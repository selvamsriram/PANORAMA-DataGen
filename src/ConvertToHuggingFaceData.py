import pandas as pd
import json
from datasets import Dataset, Features, Value

# --- Configuration ---
# !!! IMPORTANT: Replace this with the actual path to your TSV file in Google Drive !!!
DRIVE_FILE_PATH = '/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/Azure_Synthetic_Data_10K.pretraining.tsv'

# --- Load Data from TSV ---
try:
    # Load the TSV file into a pandas DataFrame
    # Adjust 'sep' if your file uses a different delimiter
    df = pd.read_csv(DRIVE_FILE_PATH, sep='\t')
    print(f"Successfully loaded data from {DRIVE_FILE_PATH}")
    print("Original DataFrame head:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File not found at {DRIVE_FILE_PATH}")
    print("Please ensure the path is correct and the file exists in your Google Drive.")
    # Handle the error appropriately
    raise
except Exception as e:
    print(f"Error reading TSV file: {e}")
    # Handle other potential errors during file reading
    raise

# --- Parse JSON in 'Text' Column ---
# Define a function to safely parse JSON, handling potential errors
def safe_json_loads(text_json):
    try:
        # Check if the input is already not a string (e.g., NaN, None)
        if not isinstance(text_json, str):
            return None # Or handle as appropriate (e.g., return an empty string)
        return json.loads(text_json)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON: {text_json[:50]}...") # Log problematic entry
        return None # Return None or some default value for invalid JSON
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
        return None

# Apply the safe JSON parsing function to the 'Text' column
# This creates a new column 'parsed_text'
df['parsed_text'] = df['Text'].apply(safe_json_loads)

# Handle rows where JSON parsing failed (resulted in None)
# Option 1: Drop rows with parsing errors
# df = df.dropna(subset=['parsed_text'])
# Option 2: Fill with a default value (e.g., empty string)
df['parsed_text'] = df['parsed_text'].fillna('')

# Select only the necessary column ('parsed_text') and rename it to 'text'
# to match the expected format for the Hugging Face Dataset
df_processed = df[['parsed_text']].rename(columns={'parsed_text': 'text'})

print("\nProcessed DataFrame head (after JSON parsing and renaming):")
print(df_processed.head())

# --- Convert to Hugging Face Dataset ---
# Define the features of the dataset
features = Features({
    'text': Value(dtype='string', id=None),
    # Add other columns here if needed, matching the DataFrame
})

# Create the Hugging Face Dataset object from the pandas DataFrame
hf_dataset = Dataset.from_pandas(df_processed, features=features)

print("\nCreated Hugging Face Dataset:")
print(hf_dataset)

# # --- Apply Formatting Function ---
# # Ensure 'tokenizer' and 'EOS_TOKEN' are defined before this step
# # For example:
# # from transformers import AutoTokenizer
# # tokenizer = AutoTokenizer.from_pretrained("gpt2") # Or your specific tokenizer
# # EOS_TOKEN = tokenizer.eos_token

# # Check if EOS_TOKEN is defined (replace with your actual check/definition)
# if 'EOS_TOKEN' not in globals():
#     print("\nError: 'EOS_TOKEN' is not defined. Please define it before running the formatting function.")
#     # Define a dummy token for demonstration if needed, but use your actual one
#     # EOS_TOKEN = "<|endoftext|>"
#     raise NameError("EOS_TOKEN is not defined")


# def formatting_prompts_func(examples):
#     """Appends the EOS_TOKEN to each text example."""
#     return { "text" : [example + EOS_TOKEN for example in examples["text"]] }

# # Apply the formatting function to the dataset
# dataset_formatted = hf_dataset.map(formatting_prompts_func, batched=True)

# print("\nFormatted Dataset head:")
# # Print the first few examples of the formatted text
# for i in range(min(5, len(dataset_formatted))):
#      print(f"Example {i}: {dataset_formatted[i]['text']}")

# # Now 'dataset_formatted' contains your data loaded from Google Drive,
# # processed, and formatted, ready for use.
