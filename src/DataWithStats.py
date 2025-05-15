import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# WordCloud is removed
from collections import Counter
import os
import re # For more advanced PII detection if needed
import numpy as np # For potential numerical operations

# --- Configuration ---
JSONL_FILE_PATH = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/Azure_Synthetic_Data_10K.processed.jsonl"  # <--- !!! SET YOUR FILE PATH HERE !!!
OUTPUT_PLOT_DIR = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/Data_Analysis_Results/" # Directory to save plots
TOP_N_CATEGORIES = 15 # For plots with many categories

# --- 1. Color Theme and Style Setup ---
sns.set_style("whitegrid")
try:
    categorical_palette = sns.color_palette("tab20", 20)
except ValueError:
    categorical_palette = plt.cm.get_cmap('tab20').colors

sequential_palette_name = "viridis" # Used for histograms

# --- 2. Data Loading and Initial Parsing ---

def load_jsonl_data(file_path):
    """Loads data from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number + 1}: {e}")
    return records

def robust_float_conversion(value_str, default_val=0.0):
    """Converts a string (potentially with currency symbols) to float."""
    if value_str is None:
        return default_val
    try:
        # Remove anything that's not a digit or a decimal point
        cleaned_str = re.sub(r'[^\d.]', '', str(value_str))
        if cleaned_str: # Ensure not empty after cleaning
            return float(cleaned_str)
        return default_val
    except ValueError:
        return default_val

def extract_features(record):
    """
    Extracts PII features and synthetic content details from a single record.
    """
    pii_input = record.get("synthetic_pii_input", {})
    synthetic_data = record.get("SyntheticTrainingData", {}).get("content_pairs", [])
    real_person_input = record.get("real_person_text_input", "")
    error_info = record.get("error")

    extracted_rows = []

    # Robust float conversion for currency fields
    net_worth = robust_float_conversion(pii_input.get("Net Worth"))
    annual_salary = robust_float_conversion(pii_input.get("Annual Salary"))

    for content_pair_idx, content_pair in enumerate(synthetic_data):
        row = {
            "record_line_number": record.get("line_number", -1),
            "pii_unique_id": pii_input.get("Unique ID"),
            "content_pair_index": content_pair_idx,
            "pii_locale": pii_input.get("Locale"),
            "pii_first_name_val": pii_input.get("First Name"), # Store original PII for reference
            "pii_last_name_val": pii_input.get("Last Name"),
            "pii_gender": pii_input.get("Gender"),
            "pii_age": pii_input.get("Age"),
            "pii_nationality": pii_input.get("Nationality"),
            "pii_marital_status": pii_input.get("Marital Status"),
            "pii_children_count": pii_input.get("Children Count"),
            "pii_education_info": pii_input.get("Education Info"),
            "pii_finance_status": pii_input.get("Finance Status"),
            "pii_net_worth": net_worth,
            "pii_employer": pii_input.get("Employer"),
            "pii_job_title": pii_input.get("Job Title"),
            "pii_annual_salary": annual_salary,
            "pii_credit_score": pii_input.get("Credit Score"),
            "pii_blood_type": pii_input.get("Blood Type"),
            "pii_birth_city": pii_input.get("Birth City"),
            "pii_has_allergies": pii_input.get("Allergies", "None").lower() != "none",
            "pii_has_disability": pii_input.get("Disability", "None").lower() != "none",
            "content_type": content_pair.get("ContentType"),
            "synthetic_text": content_pair.get("Text", ""),
            "synthetic_text_length": len(content_pair.get("Text", "")),
            "synthetic_word_count": len(content_pair.get("Text", "").split()),
            "real_person_text_length": len(real_person_input),
            "generation_error": error_info is not None,
            "error_details": str(error_info) if error_info else None,
        }

        mentioned_pii_fields = []
        text_lower = row["synthetic_text"].lower()

        # Comprehensive list of PII fields to check for textual mentions
        pii_values_to_check = {
            "First Name": str(pii_input.get("First Name","")).lower(),
            "Last Name": str(pii_input.get("Last Name","")).lower(),
            "Father's Name": str(pii_input.get("Father's Name","")).lower(),
            "Mother's Name": str(pii_input.get("Mother's Name","")).lower(),
            "Gender": str(pii_input.get("Gender","")).lower(),
            "Age": str(pii_input.get("Age","")).lower(), # Check as string
            "Nationality": str(pii_input.get("Nationality","")).lower(),
            "Marital Status": str(pii_input.get("Marital Status","")).lower(),
            "Spouse Name": str(pii_input.get("Spouse Name","")).lower().split()[0], # Check as string
            #"Children Count": str(pii_input.get("Children Count","")).lower(), # Check as string
            "National ID": re.sub(r'[^a-z0-9]', '', str(pii_input.get("National ID","")).lower()),
            "Passport Number": re.sub(r'[^a-z0-9]', '', str(pii_input.get("Passport Number","")).lower()),
            "Driver's License": re.sub(r'[^a-z0-9]', '', str(pii_input.get("Driver's License","")).lower()),
            "Phone Number": re.sub(r'\D', '', str(pii_input.get("Phone Number",""))),
            "Work Phone": re.sub(r'\D', '', str(pii_input.get("Work Phone",""))),
            "Address": str(pii_input.get("Address","")).lower(), # Address matching needs care
            "Email Address": str(pii_input.get("Email Address","")).lower(),
            "Work Email": str(pii_input.get("Work Email","")).lower(),
            "Birth Date": str(pii_input.get("Birth Date","")).lower(),
            "Birth City": str(pii_input.get("Birth City","")).lower(),
            "Education Info": str(pii_input.get("Education Info","")).lower(),
            "Finance Status": str(pii_input.get("Finance Status","")).lower(),
            "Employer": str(pii_input.get("Employer","")).lower(),
            "Job Title": str(pii_input.get("Job Title","")).lower(),
            "Credit Score": str(pii_input.get("Credit Score","")).lower(), # Check as string
            "Blood Type": str(pii_input.get("Blood Type","")).lower(),
            "Emergency Contact Name": str(pii_input.get("Emergency Contact Name","")).lower(),
            "Emergency Contact Phone": re.sub(r'\D', '', str(pii_input.get("Emergency Contact Phone",""))),
        }

        # Allergies and Disability (only if not "None")
        allergies = str(pii_input.get("Allergies","")).lower()
        if allergies != "none" and allergies:
            pii_values_to_check["Allergies"] = allergies
        disability = str(pii_input.get("Disability","")).lower()
        if disability != "none" and disability:
            pii_values_to_check["Disability"] = disability

        # Normalization for text matching
        text_for_phone_check = re.sub(r'\D', '', text_lower)
        text_for_alphanum_id_check = re.sub(r'[^a-z0-9]', '', text_lower)

        for field, value in pii_values_to_check.items():
            # Adjusted length check: allow short purely numeric strings, require len >=3 for others
            if value and ( (value.isdigit() and len(value) >= 1) or (not value.isdigit() and len(value) >= 3) ):
                if field in ["Phone Number", "Work Phone", "Emergency Contact Phone"]:
                    if value and value in text_for_phone_check:
                        mentioned_pii_fields.append(field)
                elif field in ["National ID", "Passport Number", "Driver's License"]:
                    if value and value in text_for_alphanum_id_check:
                        mentioned_pii_fields.append(field)
                elif field == "Address":
                    addr_parts = [part.strip() for part in value.split(',') if len(part.strip()) >= 3]
                    addr_parts.extend([part.strip() for part in value.split() if len(part.strip()) >= 3 and part.strip().lower() not in ["mount", "road", "street", "st", "rd", "ave", "avenue", "ln", "lane", "dr", "drive", "blvd", "boulevard", "ct", "court", "pl", "place"]])
                    if any(part in text_lower for part in addr_parts if part): # Ensure part is not empty
                        mentioned_pii_fields.append(field)
                elif value in text_lower:
                    mentioned_pii_fields.append(field)

        # Check for Social Media Handles
        social_media_handles = pii_input.get("Social Media Handles", {})
        if isinstance(social_media_handles, dict):
            for platform, handle in social_media_handles.items():
                normalized_handle = str(handle).lower()
                # Handles can be short, but check if it's non-empty
                if normalized_handle and normalized_handle in text_lower:
                    mentioned_pii_fields.append("Social Media Handle") # Generic field name for any handle mention
                    # To be more specific: mentioned_pii_fields.append(f"Social Media Handle ({platform})")


        row["mentioned_pii_fields"] = list(set(mentioned_pii_fields))
        row["num_mentioned_pii_fields"] = len(row["mentioned_pii_fields"])
        row["any_pii_mentioned"] = row["num_mentioned_pii_fields"] > 0
        extracted_rows.append(row)
    return extracted_rows

# --- 3. Plotting Utilities (Heatmap and Wordcloud removed) ---

def create_plot_dir(dir_name=OUTPUT_PLOT_DIR):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def plot_distribution(data_series, title, xlabel, ylabel="Frequency", kind='bar', top_n=None, output_dir=OUTPUT_PLOT_DIR, palette=None, rotate_labels=False):
    plt.figure(figsize=(12, 7))
    current_palette = palette if palette is not None else categorical_palette

    if kind == 'hist':
        hist_palette_name = palette if palette else sequential_palette_name # Use string name for seaborn
        # If palette is a list (e.g. a single color), use color= argument
        if isinstance(hist_palette_name, list):
             sns.histplot(data_series.dropna(), kde=True, color=hist_palette_name[0])
        else: # It's a string name of a palette
             sns.histplot(data_series.dropna(), kde=True, palette=hist_palette_name)

    elif kind == 'bar':
        counts = data_series.value_counts()
        if top_n:
            counts = counts.head(top_n)
        bar_palette = current_palette
        if isinstance(current_palette, list) and len(counts) > 0 : # Check if counts is not empty
            if len(counts) > len(current_palette):
                bar_palette = [current_palette[i % len(current_palette)] for i in range(len(counts))]
            elif isinstance(current_palette, list):
                bar_palette = current_palette[:len(counts)]
        elif not isinstance(current_palette, list) and len(counts) > 0: # It's a string name
            bar_palette = sns.color_palette(current_palette, len(counts))


        if not counts.empty: # Only plot if there are counts
            sns.barplot(x=counts.index, y=counts.values, palette=bar_palette)
            if rotate_labels or (len(counts.index) > 5 and counts.index.astype(str).str.len().max() > 10):
                 plt.xticks(rotation=45, ha="right")
        else:
            plt.text(0.5, 0.5, "No data to display", ha='center', va='center')


    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}.png"), dpi=300)
    plt.close()


def plot_stacked_bar(df_grouped, title, xlabel, ylabel, output_dir=OUTPUT_PLOT_DIR, palette=None):
    current_palette = palette if palette is not None else categorical_palette
    num_bars_per_group = len(df_grouped.columns)
    
    bar_palette = current_palette
    if isinstance(current_palette, list):
        if num_bars_per_group > len(current_palette):
            bar_palette = [current_palette[i % len(current_palette)] for i in range(num_bars_per_group)]
        else:
            bar_palette = current_palette[:num_bars_per_group]
    else: # It's a string name for a seaborn palette
        bar_palette = sns.color_palette(current_palette, num_bars_per_group)

    if not df_grouped.empty:
        df_grouped.plot(kind='bar', stacked=True, figsize=(14, 8), color=bar_palette)
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title=df_grouped.columns.name if hasattr(df_grouped.columns, 'name') else 'Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}_stacked.png"), dpi=300)
    else:
        print(f"Skipping stacked bar plot '{title}' as data is empty.")
    plt.close()

def plot_boolean_comparison(df, bool_column_name, compare_column_name, title, output_dir=OUTPUT_PLOT_DIR, top_n=None, palette=None):
    current_palette = palette if palette is not None else categorical_palette
    plt.figure(figsize=(14, 8))
    counts = df.groupby(bool_column_name)[compare_column_name].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    
    if counts.empty:
        print(f"Skipping boolean comparison plot '{title}' as no data after grouping.")
        plt.close()
        return
        
    if top_n:
        top_categories = df[compare_column_name].value_counts().nlargest(top_n).index
        counts = counts[counts[compare_column_name].isin(top_categories)]
    
    if counts.empty:
        print(f"Skipping boolean comparison plot '{title}' as no data after top_n filtering.")
        plt.close()
        return

    hue_palette = current_palette[:2] if isinstance(current_palette, list) else sns.color_palette(current_palette, 2)

    sns.barplot(x=compare_column_name, y='percentage', hue=bool_column_name, data=counts, palette=hue_palette)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(compare_column_name, fontsize=12)
    plt.ylabel('Percentage within Group (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=bool_column_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}.png"), dpi=300)
    plt.close()

def plot_pii_field_by_content_type(df_exploded_pii, title, output_dir=OUTPUT_PLOT_DIR, palette=None):
    current_palette = palette if palette is not None else categorical_palette
    if df_exploded_pii.empty or 'mentioned_pii_fields' not in df_exploded_pii.columns or 'content_type' not in df_exploded_pii.columns:
        print("Not enough data to plot PII field by content type.")
        return

    pii_content_counts = df_exploded_pii.groupby(['mentioned_pii_fields', 'content_type']).size().unstack(fill_value=0)

    if pii_content_counts.empty:
        print("No PII mentions found to plot by content type.")
        return

    num_pii_fields = len(pii_content_counts.index)
    num_content_types = len(pii_content_counts.columns)
    
    if num_content_types == 0:
        print("No content types found for PII field breakdown plot.")
        return

    bar_palette = current_palette
    if isinstance(current_palette, list):
        if num_content_types > len(current_palette):
            bar_palette = [current_palette[i % len(current_palette)] for i in range(num_content_types)]
        else:
            bar_palette = current_palette[:num_content_types]
    else:
        bar_palette = sns.color_palette(current_palette, num_content_types)
    
    fig_width = max(15, num_pii_fields * 1.0 + num_content_types * 0.5)
    fig_height = 8 + num_pii_fields * 0.15

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    pii_content_counts.plot(kind='bar', width=0.75, color=bar_palette, ax=ax)

    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("PII Field Mentioned", fontsize=14)
    ax.set_ylabel("Number of Mentions", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend across the bottom of the full figure
    fig.legend(
        title="Content Type",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=num_content_types if num_content_types < 5 else 5,
        fontsize=10,
        title_fontsize=11
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])  # Leave space at bottom for legend
    fig.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}.png"), dpi=300)
    plt.close()

# def plot_pii_field_by_content_type(df_exploded_pii, title, output_dir=OUTPUT_PLOT_DIR, palette=None):
#     current_palette = palette if palette is not None else categorical_palette
#     if df_exploded_pii.empty or 'mentioned_pii_fields' not in df_exploded_pii.columns or 'content_type' not in df_exploded_pii.columns:
#         print("Not enough data to plot PII field by content type.")
#         return

#     pii_content_counts = df_exploded_pii.groupby(['mentioned_pii_fields', 'content_type']).size().unstack(fill_value=0)

#     if pii_content_counts.empty:
#         print("No PII mentions found to plot by content type.")
#         return

#     num_pii_fields = len(pii_content_counts.index)
#     num_content_types = len(pii_content_counts.columns)
    
#     if num_content_types == 0:
#         print("No content types found for PII field breakdown plot.")
#         return

#     bar_palette = current_palette
#     if isinstance(current_palette, list):
#         if num_content_types > len(current_palette):
#             bar_palette = [current_palette[i % len(current_palette)] for i in range(num_content_types)]
#         else:
#             bar_palette = current_palette[:num_content_types]
#     else:
#         bar_palette = sns.color_palette(current_palette, num_content_types)
    
#     fig_width = max(15, num_pii_fields * 1.0 + num_content_types * 0.5) # Adjusted dynamic width
#     fig_height = 8 + num_pii_fields * 0.15 # Adjusted dynamic height

#     pii_content_counts.plot(kind='bar', figsize=(fig_width, fig_height), width=0.75, color=bar_palette)
    
#     plt.title(title, fontsize=18, pad=20)
#     plt.xlabel("PII Field Mentioned", fontsize=14)
#     plt.ylabel("Number of Mentions", fontsize=14)
#     plt.xticks(rotation=60, ha="right", fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.legend(title="Content Type", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
#     #plt.legend(title="Content Type", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
#     plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust rect for legend
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}.png"), dpi=300)
#     plt.close()


# --- 4. Main Analysis and Plotting Logic ---
def main():
    print(f"Loading data from {JSONL_FILE_PATH}...")
    raw_records = load_jsonl_data(JSONL_FILE_PATH)
    if not raw_records:
        print("No records loaded. Exiting.")
        return
    print(f"Loaded {len(raw_records)} raw records.")

    plot_dir = create_plot_dir()
    print(f"Plots will be saved to: {plot_dir}")

    all_extracted_data = []
    for record in raw_records:
        all_extracted_data.extend(extract_features(record))

    df = pd.DataFrame(all_extracted_data)
    print(f"Created DataFrame with {len(df)} rows (one per content_pair) and {len(df.columns)} columns.")
    
    if df.empty:
        print("DataFrame is empty after feature extraction. Exiting.")
        return

    print("\n--- Basic Statistics ---")
    print(f"Total number of synthetic content pieces: {len(df)}")
    unique_pii_profiles_count = df['pii_unique_id'].nunique()
    print(f"Number of unique PII profiles processed: {unique_pii_profiles_count}")
    if unique_pii_profiles_count > 0:
        print(f"Average synthetic texts per PII profile: {len(df) / unique_pii_profiles_count:.2f}")
    if 'generation_error' in df.columns:
        print(f"Number of records with generation errors: {df['generation_error'].sum()}")

    pii_df = df.drop_duplicates(subset=['pii_unique_id']).reset_index(drop=True)
    print(f"Unique PII profiles for analysis: {len(pii_df)}")

    # --- PII Input Analysis ---
    print("\n--- Analyzing PII Input Characteristics ---")
    if not pii_df.empty:
        plot_distribution(pii_df['pii_age'].dropna(), "Age Distribution of PII Profiles", "Age", kind='hist', output_dir=plot_dir, palette=[categorical_palette[0]])
        plot_distribution(pii_df['pii_gender'].dropna(), "Gender Distribution of PII Profiles", "Gender", output_dir=plot_dir)
        plot_distribution(pii_df['pii_nationality'].dropna(), "Nationality Distribution (Top N)", "Nationality", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
        plot_distribution(pii_df['pii_job_title'].dropna(), "Job Title Distribution (Top N)", "Job Title", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
        plot_distribution(pii_df['pii_education_info'].dropna(), "Education Level of PII Profiles", "Education Level", output_dir=plot_dir, rotate_labels=True)
        plot_distribution(pii_df['pii_locale'].dropna(), "Locale Distribution of PII Profiles", "Locale", output_dir=plot_dir, rotate_labels=True)
        plot_distribution(pii_df['pii_marital_status'].dropna(), "Marital Status of PII Profiles", "Marital Status", output_dir=plot_dir, rotate_labels=True)
        plot_distribution(pii_df['pii_finance_status'].dropna(), "Financial Status of PII Profiles", "Financial Status", output_dir=plot_dir, rotate_labels=True)
        plot_distribution(pii_df['pii_net_worth'].dropna(), "Net Worth Distribution of PII Profiles", "Net Worth ($)", kind='hist', output_dir=plot_dir, palette=[categorical_palette[1]])
        plot_distribution(pii_df['pii_annual_salary'].dropna(), "Annual Salary Distribution of PII Profiles", "Annual Salary ($)", kind='hist', output_dir=plot_dir, palette=[categorical_palette[2]])

    # --- Synthetic Content Analysis ---
    print("\n--- Analyzing Synthetic Content Characteristics ---")
    plot_distribution(df['content_type'].dropna(), "Distribution of Synthetic Content Types", "Content Type", output_dir=plot_dir, rotate_labels=True)
    plot_distribution(df['synthetic_text_length'].dropna(), "Distribution of Synthetic Text Lengths", "Text Length (chars)", kind='hist', output_dir=plot_dir, palette=[categorical_palette[3]])
    plot_distribution(df['synthetic_word_count'].dropna(), "Distribution of Synthetic Word Counts", "Word Count", kind='hist', output_dir=plot_dir, palette=[categorical_palette[4]])
    # Wordcloud call removed

    # --- PII Memorization/Mention Analysis ---
    print("\n--- Analyzing PII Mentions in Synthetic Content ---")
    print(f"Total synthetic texts with at least one PII mention: {df['any_pii_mentioned'].sum()} ({df['any_pii_mentioned'].mean()*100:.2f}%)")
    plot_distribution(df['num_mentioned_pii_fields'].dropna(), "Distribution of Num PII Fields Mentioned per Text", "Number of PII Fields", kind='hist', output_dir=plot_dir, palette=[categorical_palette[5]])

    all_mentioned_fields_list = [field for sublist in df['mentioned_pii_fields'] for field in sublist]
    if all_mentioned_fields_list:
        mentioned_field_counts = Counter(all_mentioned_fields_list)
        mf_series = pd.Series(mentioned_field_counts).sort_values(ascending=False)
        plot_distribution(mf_series, "Most Frequently Mentioned PII Fields (Top N)", "PII Field", "Frequency", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
    else:
        print("No PII fields were detected as mentioned across all texts.")

    # --- NEW CHART: PII Field Mentions by Content Type ---
    print("\n--- Generating PII Field Mentions by Content Type Chart ---")
    if 'mentioned_pii_fields' in df.columns and df['any_pii_mentioned'].sum() > 0 :
        df_exploded_pii = df[df['any_pii_mentioned']].explode('mentioned_pii_fields')
        df_exploded_pii.dropna(subset=['mentioned_pii_fields'], inplace=True)
        if not df_exploded_pii.empty:
             plot_pii_field_by_content_type(df_exploded_pii,
                                       "Breakdown of Sensitive Field Mentions by Content Type",
                                       output_dir=plot_dir)
        else:
            print("No PII mentions to analyze for the breakdown chart after exploding and NaN removal.")
    else:
        print("Skipping PII Field Mentions by Content Type chart: No PII mentions found or 'mentioned_pii_fields' column is missing.")


    # --- Cross-Analysis: PII Input vs. PII Mentions & Content Type ---
    print("\n--- Cross-Analysis: PII Input vs. Output ---")
    if 'any_pii_mentioned' in df.columns and 'content_type' in df.columns:
        pii_mention_by_content_type = df.groupby('content_type')['any_pii_mentioned'].mean().mul(100).sort_values(ascending=False)
        if not pii_mention_by_content_type.empty:
            plt.figure(figsize=(12, 7))
            current_pal = categorical_palette[:len(pii_mention_by_content_type)] if isinstance(categorical_palette, list) else categorical_palette
            sns.barplot(x=pii_mention_by_content_type.index, y=pii_mention_by_content_type.values, palette=current_pal)
            plt.title("Percentage of Texts with PII Mentions by Content Type", fontsize=16, pad=20)
            plt.xlabel("Content Type", fontsize=12)
            plt.ylabel("Texts with PII Mentions (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "pii_mentions_by_content_type.png"), dpi=300)
            plt.close()

    if 'pii_age' in df.columns and 'num_mentioned_pii_fields' in df.columns and df['pii_age'].notna().any():
        max_age_val = df['pii_age'].max() if df['pii_age'].notna().any() else 75 # Default max if all NaN
        bins = [0, 18, 30, 45, 60, 75, max(max_age_val + 1, 76.0)]
        labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
        
        # Adjust bins and labels if max_age is lower than the predefined categories
        final_bins = [b for b in bins if b <= max_age_val + 1]
        if final_bins[-1] <= max_age_val : # Ensure the last bin covers max_age_val
             if final_bins[-1] < max_age_val +1 and len(final_bins) < len(bins): # Extend last bin if cut short
                  final_bins[-1] = max_age_val + 1
             elif len(final_bins) == len(bins) and final_bins[-1] < max_age_val +1 : # max_age is beyond original scheme
                  pass # final_bins already includes max_age_val+1
             elif final_bins[-1] > max_age_val + 1 and len(final_bins)>1: # Last bin too large, adjust
                  final_bins[-1] = max_age_val + 1


        # Ensure there's at least one bin beyond 0
        if not final_bins or (len(final_bins) == 1 and final_bins[0] == 0):
            final_bins = [0, max(max_age_val + 1, 1.0)] # Default to one bin if all ages are 0 or very similar
        
        final_labels = labels[:len(final_bins)-1]

        if final_bins and final_labels and len(final_bins) == len(final_labels) + 1:
            df['pii_age_group'] = pd.cut(df['pii_age'], bins=final_bins, labels=final_labels, right=False, include_lowest=True)
            avg_mentions_by_age = df.groupby('pii_age_group', observed=False)['num_mentioned_pii_fields'].mean().sort_values(ascending=False)
            if not avg_mentions_by_age.empty:
                plt.figure(figsize=(12, 7))
                current_pal = categorical_palette[:len(avg_mentions_by_age)] if isinstance(categorical_palette, list) else categorical_palette
                sns.barplot(x=avg_mentions_by_age.index, y=avg_mentions_by_age.values, palette=current_pal)
                plt.title("Average Number of PII Fields Mentioned by PII Age Group", fontsize=16, pad=20)
                plt.xlabel("PII Age Group", fontsize=12)
                plt.ylabel("Avg. Num PII Fields Mentioned", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "avg_pii_mentions_by_age_group.png"), dpi=300)
                plt.close()
        else:
            print(f"Skipping PII age group plot due to binning issues. Bins: {final_bins}, Labels: {final_labels}")


    if 'any_pii_mentioned' in df.columns and 'content_type' in df.columns:
        plot_boolean_comparison(df, 'any_pii_mentioned', 'content_type',
                                "Content Type Dist - PII Mentioned vs. Not Mentioned",
                                output_dir=plot_dir, top_n=TOP_N_CATEGORIES)

    if 'pii_job_title' in df.columns and 'any_pii_mentioned' in df.columns:
        # Ensure pii_job_title is not all NaN
        if df['pii_job_title'].notna().any():
            top_job_titles = df['pii_job_title'].value_counts().nlargest(TOP_N_CATEGORIES).index
            df_top_jobs = df[df['pii_job_title'].isin(top_job_titles)]
            if not df_top_jobs.empty:
                mentions_by_job = df_top_jobs.groupby('pii_job_title')['any_pii_mentioned'].mean().mul(100).sort_values(ascending=False)
                if not mentions_by_job.empty:
                    plt.figure(figsize=(14, 8))
                    current_pal = categorical_palette[:len(mentions_by_job)] if isinstance(categorical_palette, list) else categorical_palette
                    sns.barplot(x=mentions_by_job.index, y=mentions_by_job.values, palette=current_pal)
                    plt.title(f"PII Mention Rate by PII Job Title (Top {len(mentions_by_job)})", fontsize=16, pad=20)
                    plt.xlabel("PII Job Title", fontsize=12)
                    plt.ylabel("Texts with PII Mentions (%)", fontsize=12)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, "pii_mentions_by_top_job_titles.png"), dpi=300)
                    plt.close()
        else:
            print("Skipping PII mention rate by job title as 'pii_job_title' column has no valid data.")


    if 'pii_job_title' in df.columns and 'content_type' in df.columns:
        if df['pii_job_title'].notna().any() and df['content_type'].notna().any():
            num_top_jobs_stacked = 8
            top_jobs_for_stacked = df['pii_job_title'].value_counts().nlargest(num_top_jobs_stacked).index
            df_top_jobs_stacked = df[df['pii_job_title'].isin(top_jobs_for_stacked)]
            if not df_top_jobs_stacked.empty:
                job_content_type_dist = df_top_jobs_stacked.groupby('pii_job_title')['content_type'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
                if not job_content_type_dist.empty:
                    plot_stacked_bar(job_content_type_dist,
                                     f"Content Type Distribution per PII Job Title (Top {len(job_content_type_dist)})",
                                     "PII Job Title", "Percentage of Content Types (%)", output_dir=plot_dir)
        else:
            print("Skipping content type distribution by job title due to missing data in relevant columns.")

    # Correlation Heatmap call removed

    if 'generation_error' in df.columns and df['generation_error'].any():
        print("\n--- Analyzing Generation Errors ---")
        error_pii_profiles = df[df['generation_error']].drop_duplicates(subset=['pii_unique_id'])
        print(f"Total unique PII profiles associated with generation errors: {len(error_pii_profiles)}")
        if not error_pii_profiles.empty and error_pii_profiles['pii_job_title'].notna().any():
            plot_distribution(error_pii_profiles['pii_job_title'].dropna(), "Job Titles in PII Profiles with Errors (Top N)", "Job Title", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
        elif not error_pii_profiles.empty:
            print("Cannot plot job titles for error profiles as 'pii_job_title' column has no valid data for these errors.")


    print(f"\nAnalysis complete. All plots saved to '{plot_dir}'.")

if __name__ == "__main__":
    if not os.path.exists(JSONL_FILE_PATH):
        print(f"ERROR: The data file was not found at '{JSONL_FILE_PATH}'.")
        print("Please ensure the JSONL_FILE_PATH variable in the script is set correctly.")
    else:
        main()