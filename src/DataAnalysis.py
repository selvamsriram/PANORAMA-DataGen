import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
import re # For more advanced PII detection if needed

# --- Configuration ---
JSONL_FILE_PATH = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/Azure_Synthetic_Data_10K.processed.jsonl"  # <--- !!! SET YOUR FILE PATH HERE !!!
OUTPUT_PLOT_DIR = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/Data_Analysis_Results/" # Directory to save plots
TOP_N_CATEGORIES = 15 # For plots with many categories

# --- 1. Data Loading and Initial Parsing ---

def load_jsonl_data(file_path):
    """Loads data from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number + 1}: {e}")
                # Optionally, skip the line or handle the error as needed
    return records

def extract_features(record):
    """
    Extracts PII features and synthetic content details from a single record.
    Returns a list of dictionaries, where each dictionary represents a single content_pair
    augmented with PII info.
    """
    pii_input = record.get("synthetic_pii_input", {})
    synthetic_data = record.get("SyntheticTrainingData", {}).get("content_pairs", [])
    real_person_input = record.get("real_person_text_input", "")
    error_info = record.get("error")

    extracted_rows = []

    for content_pair in synthetic_data:
        row = {
            # PII Features
            "pii_unique_id": pii_input.get("Unique ID"),
            "pii_locale": pii_input.get("Locale"),
            "pii_first_name": pii_input.get("First Name"),
            "pii_last_name": pii_input.get("Last Name"),
            "pii_gender": pii_input.get("Gender"),
            "pii_age": pii_input.get("Age"),
            "pii_nationality": pii_input.get("Nationality"),
            "pii_marital_status": pii_input.get("Marital Status"),
            "pii_children_count": pii_input.get("Children Count"),
            "pii_education_info": pii_input.get("Education Info"),
            "pii_finance_status": pii_input.get("Finance Status"),
            #"pii_net_worth": float(str(pii_input.get("Net Worth", "0")).replace("$", "").replace(",", "")) if pii_input.get("Net Worth") else 0,
            "pii_employer": pii_input.get("Employer"),
            "pii_job_title": pii_input.get("Job Title"),
            #"pii_annual_salary": float(str(pii_input.get("Annual Salary", "0")).replace("$", "").replace(",", "")) if pii_input.get("Annual Salary") else 0,
            "pii_credit_score": pii_input.get("Credit Score"),
            "pii_blood_type": pii_input.get("Blood Type"),
            "pii_birth_city": pii_input.get("Birth City"),
            "pii_has_allergies": pii_input.get("Allergies", "None").lower() != "none",
            "pii_has_disability": pii_input.get("Disability", "None").lower() != "none",

            # Synthetic Content Features
            "content_type": content_pair.get("ContentType"),
            "synthetic_text": content_pair.get("Text", ""),
            "synthetic_text_length": len(content_pair.get("Text", "")),
            "synthetic_word_count": len(content_pair.get("Text", "").split()),

            # Real Person Input (for potential comparison/analysis)
            "real_person_text_length": len(real_person_input),

            # Error
            "generation_error": error_info is not None,
            "error_details": str(error_info) if error_info else None,
        }

        # PII Mention Analysis (Basic - can be significantly expanded)
        mentioned_pii_fields = []
        text_lower = row["synthetic_text"].lower()
        pii_values_to_check = {
            "First Name": str(pii_input.get("First Name","")).lower(),
            "Last Name": str(pii_input.get("Last Name","")).lower(),
            "Father's Name": str(pii_input.get("Father's Name","")).lower(),
            "Mother's Name": str(pii_input.get("Mother's Name","")).lower(),
            "Spouse Name": str(pii_input.get("Spouse Name","")).lower(),
            "National ID": str(pii_input.get("National ID","")).lower(),
            "Passport Number": str(pii_input.get("Passport Number","")).lower(),
            "Driver's License": str(pii_input.get("Driver's License","")).lower(),
            "Phone Number": str(pii_input.get("Phone Number","")).replace(" ", "").replace("(", "").replace(")", "").replace("-", ""),
            "Work Phone": str(pii_input.get("Work Phone","")).replace(" ", "").replace("(", "").replace(")", "").replace("-", ""),
            "Address": str(pii_input.get("Address","")).lower(), # Check parts of address too
            "Email Address": str(pii_input.get("Email Address","")).lower(),
            "Work Email": str(pii_input.get("Work Email","")).lower(),
            "Birth City": str(pii_input.get("Birth City","")).lower(),
            "Employer": str(pii_input.get("Employer","")).lower(),
            "Job Title": str(pii_input.get("Job Title","")).lower(), # Can be generic, be careful
        }
        # Clean the synthetic text for phone number matching
        text_for_phone_check = text_lower.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")

        for field, value in pii_values_to_check.items():
            if value and len(value) > 2 : # Avoid matching very short/common strings like "a" or "to"
                if field in ["Phone Number", "Work Phone"]:
                    if value in text_for_phone_check:
                        mentioned_pii_fields.append(field)
                elif value in text_lower:
                    mentioned_pii_fields.append(field)

        row["mentioned_pii_fields"] = list(set(mentioned_pii_fields)) # Unique fields
        row["num_mentioned_pii_fields"] = len(row["mentioned_pii_fields"])
        row["any_pii_mentioned"] = row["num_mentioned_pii_fields"] > 0

        extracted_rows.append(row)
    return extracted_rows


# --- 2. Plotting Utilities ---

def create_plot_dir(dir_name=OUTPUT_PLOT_DIR):
    """Creates the output directory for plots if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def plot_distribution(data_series, title, xlabel, ylabel="Frequency", kind='bar', top_n=None, output_dir=OUTPUT_PLOT_DIR, color='skyblue', rotate_labels=False):
    """Plots distribution of a pandas Series (categorical or numerical)."""
    plt.figure(figsize=(12, 7))
    if kind == 'hist':
        sns.histplot(data_series.dropna(), kde=True, color=color)
    elif kind == 'bar':
        counts = data_series.value_counts()
        if top_n:
            counts = counts.head(top_n)
        sns.barplot(x=counts.index, y=counts.values, palette="viridis")
        if rotate_labels or (len(counts.index) > 5 and max(len(str(x)) for x in counts.index) > 10):
             plt.xticks(rotation=45, ha="right")
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').lower()}.png"))
    plt.close()

def plot_wordcloud_from_series(text_series, title, output_dir=OUTPUT_PLOT_DIR):
    """Generates and saves a word cloud from a pandas Series of text."""
    all_text = " ".join(text_series.dropna().astype(str))
    if not all_text.strip():
        print(f"Skipping word cloud for '{title}' as no text is available.")
        return
    wordcloud = WordCloud(width=1200, height=600, background_color='white', collocations=False).generate(all_text)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').lower()}_wordcloud.png"))
    plt.close()

def plot_correlation_heatmap(df_numeric, title, output_dir=OUTPUT_PLOT_DIR):
    """Plots a correlation heatmap for numeric columns in a DataFrame."""
    if df_numeric.empty:
        print(f"Skipping correlation heatmap for '{title}' as no numeric data is available.")
        return
    corr_matrix = df_numeric.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').lower()}_correlation.png"))
    plt.close()

def plot_stacked_bar(df_grouped, title, xlabel, ylabel, output_dir=OUTPUT_PLOT_DIR, top_n_groups=None):
    """Plots a stacked bar chart from a grouped DataFrame."""
    if top_n_groups and len(df_grouped.index) > top_n_groups:
        # Select top N groups based on the sum of values if it's a multi-level group or just count
        if isinstance(df_grouped, pd.Series) and isinstance(df_grouped.index, pd.MultiIndex):
             # This case requires careful handling based on specific grouping.
             # For now, let's assume it's a simple group or pre-summed.
             summed_groups = df_grouped.groupby(level=0).sum().nlargest(top_n_groups).index
             df_grouped = df_grouped[df_grouped.index.get_level_values(0).isin(summed_groups)]
        else:
            df_grouped = df_grouped.nlargest(top_n_groups)


    df_grouped.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=df_grouped.columns.name if hasattr(df_grouped.columns, 'name') else 'Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').lower()}_stacked.png"))
    plt.close()

def plot_boolean_comparison(df, bool_column_name, compare_column_name, title, output_dir=OUTPUT_PLOT_DIR, top_n=None):
    """Compares distribution of a column based on a boolean flag."""
    plt.figure(figsize=(14, 8))
    # Calculate percentages for better comparison if counts are very different
    counts = df.groupby(bool_column_name)[compare_column_name].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    
    if top_n:
        # Get top N categories overall to focus the plot
        top_categories = df[compare_column_name].value_counts().nlargest(top_n).index
        counts = counts[counts[compare_column_name].isin(top_categories)]

    sns.barplot(x=compare_column_name, y='percentage', hue=bool_column_name, data=counts, palette="mako")
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(compare_column_name, fontsize=12)
    plt.ylabel('Percentage within Group (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=bool_column_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').lower()}.png"))
    plt.close()


# --- 3. Main Analysis and Plotting Logic ---
def main():
    """Main function to run the analysis."""
    print(f"Loading data from {JSONL_FILE_PATH}...")
    raw_records = load_jsonl_data(JSONL_FILE_PATH)
    if not raw_records:
        print("No records loaded. Exiting.")
        return

    print(f"Loaded {len(raw_records)} raw records.")

    # Create output directory for plots
    plot_dir = create_plot_dir()
    print(f"Plots will be saved to: {plot_dir}")

    # Extract features into a flat list of dictionaries
    all_extracted_data = []
    for record in raw_records:
        all_extracted_data.extend(extract_features(record))

    df = pd.DataFrame(all_extracted_data)
    print(f"Created DataFrame with {len(df)} rows (one per content_pair) and {len(df.columns)} columns.")
    print("\nDataFrame columns:", df.columns.tolist())
    print("\nSample of DataFrame head:\n", df.head())

    # --- Basic Statistics ---
    print("\n--- Basic Statistics ---")
    print(f"Total number of synthetic content pieces: {len(df)}")
    print(f"Number of unique PII profiles processed: {df['pii_unique_id'].nunique()}")
    print(f"Average synthetic texts per PII profile: {len(df) / df['pii_unique_id'].nunique():.2f}")
    if 'generation_error' in df.columns:
        print(f"Number of records with generation errors: {df['generation_error'].sum()}")


    # --- PII Input Analysis ---
    print("\n--- Analyzing PII Input Characteristics ---")
    pii_df = df.drop_duplicates(subset=['pii_unique_id']).reset_index(drop=True) # Analyze each PII profile once
    print(f"Unique PII profiles for analysis: {len(pii_df)}")

    plot_distribution(pii_df['pii_age'], "Age Distribution of PII Profiles", "Age", kind='hist', output_dir=plot_dir)
    plot_distribution(pii_df['pii_gender'], "Gender Distribution of PII Profiles", "Gender", output_dir=plot_dir)
    plot_distribution(pii_df['pii_nationality'], "Nationality Distribution of PII Profiles (Top N)", "Nationality", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
    plot_distribution(pii_df['pii_job_title'], "Job Title Distribution of PII Profiles (Top N)", "Job Title", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
    plot_distribution(pii_df['pii_education_info'], "Education Level of PII Profiles", "Education Level", output_dir=plot_dir, rotate_labels=True)
    plot_distribution(pii_df['pii_locale'], "Locale Distribution of PII Profiles", "Locale", output_dir=plot_dir, rotate_labels=True)
    plot_distribution(pii_df['pii_marital_status'], "Marital Status of PII Profiles", "Marital Status", output_dir=plot_dir, rotate_labels=True)
    plot_distribution(pii_df['pii_finance_status'], "Financial Status of PII Profiles", "Financial Status", output_dir=plot_dir, rotate_labels=True)
    #plot_distribution(pii_df['pii_net_worth'], "Net Worth Distribution of PII Profiles", "Net Worth ($)", kind='hist', output_dir=plot_dir)
    #plot_distribution(pii_df['pii_annual_salary'], "Annual Salary Distribution of PII Profiles", "Annual Salary ($)", kind='hist', output_dir=plot_dir)

    # --- Synthetic Content Analysis ---
    print("\n--- Analyzing Synthetic Content Characteristics ---")
    plot_distribution(df['content_type'], "Distribution of Synthetic Content Types", "Content Type", output_dir=plot_dir, rotate_labels=True)
    plot_distribution(df['synthetic_text_length'], "Distribution of Synthetic Text Lengths (Characters)", "Text Length", kind='hist', output_dir=plot_dir)
    plot_distribution(df['synthetic_word_count'], "Distribution of Synthetic Word Counts", "Word Count", kind='hist', output_dir=plot_dir)

    # Word cloud for all synthetic text (can be resource-intensive for very large datasets)
    if not df['synthetic_text'].empty:
        plot_wordcloud_from_series(df['synthetic_text'], "Overall Word Cloud of Synthetic Content", output_dir=plot_dir)

    # --- PII Memorization/Mention Analysis ---
    print("\n--- Analyzing PII Mentions in Synthetic Content ---")
    print(f"Total synthetic texts with at least one PII mention: {df['any_pii_mentioned'].sum()} ({df['any_pii_mentioned'].mean()*100:.2f}%)")
    plot_distribution(df['num_mentioned_pii_fields'], "Distribution of Number of PII Fields Mentioned per Text", "Number of PII Fields", kind='hist', output_dir=plot_dir)

    # What specific PII fields are mentioned most often?
    all_mentioned_fields = []
    for field_list in df['mentioned_pii_fields']:
        all_mentioned_fields.extend(field_list)
    mentioned_field_counts = Counter(all_mentioned_fields)
    if mentioned_field_counts:
        mf_series = pd.Series(mentioned_field_counts).sort_values(ascending=False)
        plot_distribution(mf_series, "Most Frequently Mentioned PII Fields in Synthetic Text (Top N)", "PII Field", "Frequency", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
    else:
        print("No PII fields were detected as mentioned.")

    # --- Cross-Analysis: PII Input vs. PII Mentions & Content Type ---
    print("\n--- Cross-Analysis: PII Input vs. Output ---")

    # PII Mentions by Content Type
    if 'any_pii_mentioned' in df.columns and 'content_type' in df.columns:
        pii_mention_by_content_type = df.groupby('content_type')['any_pii_mentioned'].mean().mul(100).sort_values(ascending=False)
        if not pii_mention_by_content_type.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=pii_mention_by_content_type.index, y=pii_mention_by_content_type.values, palette="crest")
            plt.title("Percentage of Texts with PII Mentions by Content Type", fontsize=16, pad=20)
            plt.xlabel("Content Type", fontsize=12)
            plt.ylabel("Texts with PII Mentions (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "pii_mentions_by_content_type.png"))
            plt.close()

    # Number of PII mentions by PII Age Group
    if 'pii_age' in df.columns and 'num_mentioned_pii_fields' in df.columns:
        df['pii_age_group'] = pd.cut(df['pii_age'], bins=[0, 18, 30, 45, 60, 75, 120], labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+'])
        avg_mentions_by_age = df.groupby('pii_age_group')['num_mentioned_pii_fields'].mean().sort_values(ascending=False)
        if not avg_mentions_by_age.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=avg_mentions_by_age.index, y=avg_mentions_by_age.values, palette="flare")
            plt.title("Average Number of PII Fields Mentioned by PII Age Group", fontsize=16, pad=20)
            plt.xlabel("PII Age Group", fontsize=12)
            plt.ylabel("Avg. Num PII Fields Mentioned", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "avg_pii_mentions_by_age_group.png"))
            plt.close()

    # Content Type distribution for records with PII mentions vs. those without
    if 'any_pii_mentioned' in df.columns and 'content_type' in df.columns:
        plot_boolean_comparison(df, 'any_pii_mentioned', 'content_type',
                                "Content Type Distribution: PII Mentioned vs. Not Mentioned",
                                output_dir=plot_dir, top_n=TOP_N_CATEGORIES)


    # PII Mentions by PII Job Title (Top N Job Titles)
    if 'pii_job_title' in df.columns and 'any_pii_mentioned' in df.columns:
        top_job_titles = df['pii_job_title'].value_counts().nlargest(TOP_N_CATEGORIES).index
        df_top_jobs = df[df['pii_job_title'].isin(top_job_titles)]
        mentions_by_job = df_top_jobs.groupby('pii_job_title')['any_pii_mentioned'].mean().mul(100).sort_values(ascending=False)
        if not mentions_by_job.empty:
            plt.figure(figsize=(14, 8))
            sns.barplot(x=mentions_by_job.index, y=mentions_by_job.values, palette="magma")
            plt.title(f"PII Mention Rate by PII Job Title (Top {TOP_N_CATEGORIES})", fontsize=16, pad=20)
            plt.xlabel("PII Job Title", fontsize=12)
            plt.ylabel("Texts with PII Mentions (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "pii_mentions_by_top_job_titles.png"))
            plt.close()

    # Stacked bar: Content Type generated per PII Job Title (for top N job titles)
    if 'pii_job_title' in df.columns and 'content_type' in df.columns:
        top_jobs_for_stacked = df['pii_job_title'].value_counts().nlargest(8).index # Keep this N smaller for readability
        df_top_jobs_stacked = df[df['pii_job_title'].isin(top_jobs_for_stacked)]
        job_content_type_dist = df_top_jobs_stacked.groupby('pii_job_title')['content_type'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
        if not job_content_type_dist.empty:
            plot_stacked_bar(job_content_type_dist,
                             f"Content Type Distribution per PII Job Title (Top {len(top_jobs_for_stacked)} Job Titles)",
                             "PII Job Title", "Percentage of Content Types (%)", output_dir=plot_dir)


    # Correlation Heatmap (numeric PII features and mention counts)
    numeric_cols_for_corr = [
        'pii_age', 'pii_children_count', 'pii_credit_score',
        'synthetic_text_length', 'synthetic_word_count', 'num_mentioned_pii_fields'
    ]
    # Filter out columns that might not exist or are all NaN
    existing_numeric_cols = [col for col in numeric_cols_for_corr if col in df.columns and df[col].notna().any()]
    if existing_numeric_cols:
        df_numeric_sample = df[existing_numeric_cols].dropna() # Drop rows with NaNs for correlation
        if len(df_numeric_sample) > 2: # Need at least 2 samples for correlation
             plot_correlation_heatmap(df_numeric_sample, "Correlation Heatmap of Numeric Features and PII Mentions", output_dir=plot_dir)
        else:
            print(f"Not enough data points ({len(df_numeric_sample)}) for correlation heatmap after dropping NaNs.")
    else:
        print("No suitable numeric columns found for correlation heatmap.")

    # --- Analysis of Generation Errors (if 'generation_error' column exists) ---
    if 'generation_error' in df.columns and df['generation_error'].any():
        print("\n--- Analyzing Generation Errors ---")
        error_df = df[df['generation_error']]
        print(f"Total records with errors: {len(error_df)}")

        # What kind of PII profiles lead to errors more often?
        if not error_df.empty:
            plot_distribution(error_df['pii_job_title'], "Job Titles in PII Profiles with Generation Errors (Top N)", "Job Title", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
            # Add more plots for other PII features if relevant (e.g., age, locale) for error cases.

    print(f"\nAnalysis complete. All plots saved to '{plot_dir}'.")
    print("Please review the generated plots and console output for insights.")

if __name__ == "__main__":
    # --- Safety check for file existence ---
    if not os.path.exists(JSONL_FILE_PATH):
        print(f"ERROR: The data file was not found at '{JSONL_FILE_PATH}'.")
        print("Please ensure the JSONL_FILE_PATH variable in the script is set correctly.")
        # Create a dummy file for testing if you don't have the real one yet
        # print("Creating a dummy JSONL file for demonstration purposes: dummy_data.jsonl")
        # dummy_record = {
        #     "line_number": 1,
        #     "synthetic_pii_input": {
        #         "Unique ID": "test-id-001", "Locale": "en_US", "First Name": "Jane", "Last Name": "Doe",
        #         "Gender": "Female", "Age": 30, "Nationality": "American", "Marital Status": "Single",
        #         "Children Count": 0, "Education Info": "Master's", "Finance Status": "High",
        #         "Net Worth": "$500000.00", "Employer": "Tech Corp", "Job Title": "Software Developer",
        #         "Annual Salary": "$120000.00", "Credit Score": 750, "Blood Type": "O+",
        #         "Birth City": "New York", "Allergies": "Peanuts", "Disability": "None",
        #         "Phone Number": "+1 (555) 123-4567", "Email Address": "jane.doe@example.com"
        #     },
        #     "real_person_text_input": "Some real person text here.",
        #     "error": None,
        #     "SyntheticTrainingData": {
        #         "content_pairs": [
        #             {"ContentType": "Social Media", "Text": "Loving my job at Tech Corp! #developer Jane Doe here."},
        #             {"ContentType": "Forum Post", "Text": "Any other developers in New York facing this issue?"}
        #         ]
        #     }
        # }
        # with open("dummy_data.jsonl", "w") as f:
        #     f.write(json.dumps(dummy_record) + "\n")
        # JSONL_FILE_PATH = "dummy_data.jsonl" # Temporarily switch to dummy
        # if not os.path.exists(JSONL_FILE_PATH): # Check again if dummy creation failed
        #    exit()
    else:
        main()