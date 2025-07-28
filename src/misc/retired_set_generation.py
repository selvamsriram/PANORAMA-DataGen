import pandas as pd
from datasets import load_dataset, Dataset
import os
import getpass

def process_and_join_datasets_with_pandas(
    master_panorama_name: str,
    master_panorama_plus_name: str,
    subset_dataset_name: str,
    output_dataset_name: str,
    hf_token: str = None
) -> Dataset:
    """
    Fixes IDs in a subset dataset using a master dataset and then joins
    'complete_info' from another master dataset using Pandas.
    Finally, pushes the processed dataset to the Hugging Face Hub.

    Args:
        master_panorama_name (str): The name of the main master Hugging Face dataset
                                    (e.g., "srirxml/PANORAMA").
        master_panorama_plus_name (str): The name of the additional master dataset
                                         (e.g., "srirxml/PANORAMA-Plus").
        subset_dataset_name (str): The name of the subset Hugging Face dataset
                                   (e.g., "srirxml/synthetic-pii-pretraining-n150-25x").
        output_dataset_name (str): The desired name for the fixed and joined dataset
                                   on Hugging Face Hub.
        hf_token (str, optional): Your Hugging Face API token. If not provided,
                                  it will try to use `huggingface-cli login` or prompt for it.

    Returns:
        Dataset: A new Hugging Face dataset with fixed IDs and joined 'complete_info'.
    """

    print("--- Loading Datasets ---")
    # Load master PANORAMA dataset (only 'id' and 'text' needed for ID fixing)
    print(f"Loading master dataset: {master_panorama_name}")
    master_panorama_ds = load_dataset(master_panorama_name, split='train', columns=['id', 'text'])
    master_panorama_df = master_panorama_ds.to_pandas()
    print(f"'{master_panorama_name}' loaded with {len(master_panorama_df)} rows.")

    # Load master PANORAMA-Plus dataset (only 'Unique ID' and 'complete_info' needed for join)
    print(f"Loading master dataset: {master_panorama_plus_name}")
    master_panorama_plus_ds = load_dataset(master_panorama_plus_name, split='train', columns=['Unique ID', 'complete_info'])
    master_panorama_plus_df = master_panorama_plus_ds.to_pandas()
    print(f"'{master_panorama_plus_name}' loaded with {len(master_panorama_plus_df)} rows.")

    # Load the smaller subset dataset
    print(f"Loading subset dataset: {subset_dataset_name}")
    subset_ds = load_dataset(subset_dataset_name, split='train')
    subset_df = subset_ds.to_pandas()
    print(f"'{subset_dataset_name}' loaded with {len(subset_df)} rows.")

    print("\n--- Step 1: Fix IDs in the Subset Dataset using PANORAMA ---")
    # Perform a left merge to bring in the correct 'id' from master_panorama_df
    # We join on the 'text' column, which acts as our link
    # The 'id' column from master_panorama_df will become 'id_new' (or similar)
    temp_df = pd.merge(subset_df,
                       master_panorama_df[['id', 'text']],
                       on='text',
                       how='left',
                       suffixes=('_old', '_new')) # Suffixes for overlapping column names

    # Drop the temporary and old ID columns, rename the new 'id'
    fixed_id_df = temp_df.drop(columns=['id_old']).rename(columns={'id_new': 'id'})

    #Find how many rows have NaN in the id column
    nan_id_count = fixed_id_df['id'].isna().sum()
    print(f"Number of rows with NaN in the id column: {nan_id_count}")
    
    #Write all the rows with NaN in the id column to a new tsv file
    fixed_id_df[fixed_id_df['id'].isna()].to_csv('nan_id_rows.tsv', sep='\t', index=False)

    print("\n--- Step 1.5: Recover Missing IDs using Contains Check ---")
    # Get rows with NaN IDs
    nan_id_rows = fixed_id_df[fixed_id_df['id'].isna()].copy()
    
    if len(nan_id_rows) > 0:
        # Create a unique text list from rows with NaN IDs
        unique_nan_texts = nan_id_rows['text'].unique()
        print(f"Unique texts with NaN IDs: {len(unique_nan_texts)}")
        
        # Create a map of text to id using contains check
        text_to_id_map = {}
        recovered_count = 0
        
        for subset_text in unique_nan_texts:
            # Find PANORAMA rows where the text contains the subset text
            # or where the subset text starts with the PANORAMA text
            matching_rows = master_panorama_df[
                master_panorama_df['text'].str.contains(subset_text, case=False, na=False) |
                master_panorama_df['text'].str.lower().apply(lambda x: subset_text.lower().startswith(x) if pd.notna(x) else False)
            ]
            
            if len(matching_rows) > 0:
                # Take the first match (you could implement more sophisticated matching logic here)
                text_to_id_map[subset_text] = matching_rows.iloc[0]['id']
                recovered_count += 1
        
        print(f"Recovered IDs for {recovered_count} out of {len(unique_nan_texts)} unique texts")
        
        # Apply the recovery map to the original nan_id_rows
        recovered_rows = nan_id_rows.copy()
        recovered_rows['id'] = recovered_rows['text'].map(text_to_id_map)
        
        # Count how many were actually recovered
        actually_recovered = recovered_rows['id'].notna().sum()
        still_missing = recovered_rows['id'].isna().sum()
        
        print(f"Actually recovered {actually_recovered} rows out of {len(nan_id_rows)} rows with NaN IDs")
        print(f"Still missing IDs for {still_missing} rows")
        
        # Write the still missing rows to a separate file
        if still_missing > 0:
            still_missing_rows = recovered_rows[recovered_rows['id'].isna()]
            still_missing_rows.to_csv('still_missing_id_rows.tsv', sep='\t', index=False)
            print(f"Still missing rows written to 'still_missing_id_rows.tsv'")
        
        # Merge the recovered rows with the primarily found ones
        # First, remove the original nan_id_rows from fixed_id_df
        fixed_id_df_without_nan = fixed_id_df[fixed_id_df['id'].notna()].copy()
        
        # Then add the recovered rows (only those that have valid IDs)
        recovered_with_valid_ids = recovered_rows[recovered_rows['id'].notna()].copy()
        
        # Combine the two dataframes
        final_fixed_id_df = pd.concat([fixed_id_df_without_nan, recovered_with_valid_ids], ignore_index=True)
        
        print(f"Final dataset after recovery: {len(final_fixed_id_df)} rows")
    else:
        print("No rows with NaN IDs to recover")
        final_fixed_id_df = fixed_id_df.copy()

    print("\n--- Step 2: Join 'complete_info' from PANORAMA-Plus ---")
    # Rename 'Unique ID' in PANORAMA-Plus DataFrame to 'id' to match the fixed subset_df
    master_panorama_plus_df = master_panorama_plus_df.rename(columns={'Unique ID': 'id'})

    # Perform a left join to add 'complete_info' to the fixed subset_df
    # A left join ensures all rows from subset_df are kept
    processed_df = pd.merge(final_fixed_id_df,
                            master_panorama_plus_df[['id', 'complete_info']],
                            on='id',
                            how='inner')

    # Handle cases where 'complete_info' might be missing after join (if 'id' not in PANORAMA-Plus)
    missing_info_count = processed_df['complete_info'].isna().sum()
    if missing_info_count > 0:
        print(f"Warning: {missing_info_count} rows could not find matching 'complete_info' in '{master_panorama_plus_name}'.")
        # You might want to fill these NaNs with a default value or keep them as NaN
        # For example: processed_df['complete_info'] = processed_df['complete_info'].fillna("N/A")
    else:
        print("All 'complete_info' successfully joined.")

    print("Dataset processing complete.")

    # Remove duplicate rows before converting to Hugging Face Dataset
    print(f"Removing duplicates from {len(processed_df)} rows...")
    processed_df_unique = processed_df.drop_duplicates()
    print(f"After removing duplicates: {len(processed_df_unique)} rows")
    print(f"Removed {len(processed_df) - len(processed_df_unique)} duplicate rows")

    # Convert the processed Pandas DataFrame back to a Hugging Face Dataset
    fixed_and_joined_ds = Dataset.from_pandas(processed_df_unique)
    print(f"Converted back to Hugging Face Dataset with {len(fixed_and_joined_ds)} rows and {fixed_and_joined_ds.num_columns} columns.")

    # Push to Hugging Face Hub
    print(f"\n--- Pushing fixed and joined dataset to Hugging Face Hub as: {output_dataset_name} ---")
    try:
        fixed_and_joined_ds.push_to_hub(output_dataset_name, token=hf_token)
        print(f"Successfully pushed '{output_dataset_name}' to Hugging Face Hub!")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        print("Please ensure you have authenticated with `huggingface-cli login` or provided a valid `hf_token`.")
        print("Also check your permissions for the repository name (e.g., 'your_username/your_repo_name').")

    return fixed_and_joined_ds

# --- How to use the function ---
if __name__ == "__main__":
    master_panorama_dataset_name = "srirxml/PANORAMA"
    master_panorama_plus_dataset_name = "srirxml/PANORAMA-Plus"
    subset_dataset_name = "srirxml/synthetic-pii-pretraining-n150-25x"

    # IMPORTANT: Replace 'your_username' with your actual Hugging Face username
    # or the organization name where you want to push the dataset.
    # For srirxml organization, it will be "srirxml/dataset-name"
    output_dataset_name = "srirxml/synthetic-pii-pretraining-n150-25x-with-complete-info"

    # --- Hugging Face Token Handling ---
    # Option 1: Authenticate via 'huggingface-cli login' in your terminal (recommended for long-term)
    # The script will automatically pick up the token from there.
    # Option 2: Provide token directly (less secure for production, but good for quick testing)
    # hf_token = "hf_YOUR_ACTUAL_TOKEN_HERE" # Replace with your actual write token
    hf_token = os.getenv("HF_TOKEN") # Try to get from environment variable

    if not hf_token:
        print("\nNo Hugging Face token found. Please enter your token or log in via `huggingface-cli login`.")
        try:
            hf_token = getpass.getpass("Enter your Hugging Face write token: ")
        except Exception as e:
            print(f"Could not get token interactively: {e}")
            print("Please set the HF_TOKEN environment variable or log in using 'huggingface-cli login'.")
            exit(1) # Exit if no token is available for pushing

    # Run the optimization, ID fixing, and joining
    final_dataset = process_and_join_datasets_with_pandas(
        master_panorama_name=master_panorama_dataset_name,
        master_panorama_plus_name=master_panorama_plus_dataset_name,
        subset_dataset_name=subset_dataset_name,
        output_dataset_name=output_dataset_name,
        hf_token=hf_token
    )

    print("\nScript finished.")