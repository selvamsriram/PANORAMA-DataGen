import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import re
import numpy as np

# --- Configuration ---
JSONL_FILE_PATH = "/Users/sriramselvam/Code/PANORAMA-DataGen/data/Azure_Synthetic_Data_10K.processed.jsonl"  # <--- !!! SET YOUR FILE PATH HERE !!!
OUTPUT_PLOT_DIR = "/Users/sriramselvam/Code/PANORAMA-DataGen/data/Data_Analysis_Results/" # Directory to save plots
TOP_N_CATEGORIES = 15 # For plots with many categories

# --- 1. Color Theme and Style Setup ---
sns.set_style("whitegrid")
try:
    categorical_palette = sns.color_palette("tab20", 20)
except ValueError:
    categorical_palette = plt.cm.get_cmap('tab20').colors
sequential_palette_name = "viridis"

# --- 2. Data Loading and Initial Parsing ---

def load_jsonl_data(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number + 1}: {e}")
    return records

def robust_float_conversion(value_str, default_val=0.0):
    if value_str is None: return default_val
    try:
        cleaned_str = re.sub(r'[^\d.]', '', str(value_str))
        return float(cleaned_str) if cleaned_str else default_val
    except ValueError: return default_val

def extract_features(record):
    pii_input = record.get("synthetic_pii_input", {})
    synthetic_data = record.get("SyntheticTrainingData", {}).get("content_pairs", [])
    real_person_input = record.get("real_person_text_input", "")
    error_info = record.get("error")

    extracted_rows = []
    net_worth = robust_float_conversion(pii_input.get("Net Worth"))
    annual_salary = robust_float_conversion(pii_input.get("Annual Salary"))

    for content_pair_idx, content_pair in enumerate(synthetic_data):
        current_content_type = content_pair.get("ContentType")
        current_synthetic_text = content_pair.get("Text", "")
        text_lower = current_synthetic_text.lower()

        row = {
            "record_line_number": record.get("line_number", -1),
            "pii_unique_id": pii_input.get("Unique ID"),
            "content_pair_index": content_pair_idx,
            "pii_locale": pii_input.get("Locale"),
            "pii_first_name_val": pii_input.get("First Name"),
            "pii_last_name_val": pii_input.get("Last Name"),
            "pii_gender": pii_input.get("Gender"), "pii_age": pii_input.get("Age"),
            "pii_nationality": pii_input.get("Nationality"),
            "pii_marital_status": pii_input.get("Marital Status"),
            "pii_children_count": pii_input.get("Children Count"),
            "pii_education_info": pii_input.get("Education Info"),
            "pii_finance_status": pii_input.get("Finance Status"),
            "pii_net_worth": net_worth, "pii_employer": pii_input.get("Employer"),
            "pii_job_title": pii_input.get("Job Title"),
            "pii_annual_salary": annual_salary,
            "pii_credit_score": pii_input.get("Credit Score"),
            "pii_blood_type": pii_input.get("Blood Type"),
            "pii_birth_city": pii_input.get("Birth City"),
            "pii_has_allergies": pii_input.get("Allergies", "None").lower() != "none",
            "pii_has_disability": pii_input.get("Disability", "None").lower() != "none",
            "content_type": current_content_type,
            "synthetic_text": current_synthetic_text, # Store original case text for debug
            "synthetic_text_length": len(current_synthetic_text),
            "synthetic_word_count": len(current_synthetic_text.split()),
            "real_person_text_length": len(real_person_input),
            "generation_error": error_info is not None,
            "error_details": str(error_info) if error_info else None,
            "debug_pii_mentions_details": [] # Initialize for debug info
        }

        mentioned_pii_fields_for_stats = [] # For aggregated stats (unique field names)

        # Comprehensive PII fields for textual mention checking
        pii_values_to_check = {
            # Names
            "First Name": str(pii_input.get("First Name","")).lower(),
            "Last Name": str(pii_input.get("Last Name","")).lower(),
            "Father's Name": str(pii_input.get("Father's Name","")).lower(),
            "Mother's Name": str(pii_input.get("Mother's Name","")).lower(),
            "Spouse Name": str(pii_input.get("Spouse Name","")).lower(),
            "Emergency Contact Name": str(pii_input.get("Emergency Contact Name","")).lower(),
            # Demographics & Personal Info
            "Gender": str(pii_input.get("Gender","")).lower(),
            "Age": str(pii_input.get("Age","")).lower(),
            "Nationality": str(pii_input.get("Nationality","")).lower(),
            "Marital Status": str(pii_input.get("Marital Status","")).lower(),
            "Children Count": str(pii_input.get("Children Count","")).lower(),
            "Birth Date": str(pii_input.get("Birth Date","")).lower(),
            "Birth City": str(pii_input.get("Birth City","")).lower(),
            "Blood Type": str(pii_input.get("Blood Type","")).lower(),
            # Identifiers (normalized)
            "National ID": re.sub(r'[^a-z0-9]', '', str(pii_input.get("National ID","")).lower()),
            "Passport Number": re.sub(r'[^a-z0-9]', '', str(pii_input.get("Passport Number","")).lower()),
            "Driver's License": re.sub(r'[^a-z0-9]', '', str(pii_input.get("Driver's License","")).lower()),
            # Contact (normalized for phones)
            "Phone Number": re.sub(r'\D', '', str(pii_input.get("Phone Number",""))),
            "Work Phone": re.sub(r'\D', '', str(pii_input.get("Work Phone",""))),
            "Emergency Contact Phone": re.sub(r'\D', '', str(pii_input.get("Emergency Contact Phone",""))),
            "Email Address": str(pii_input.get("Email Address","")).lower(),
            "Work Email": str(pii_input.get("Work Email","")).lower(),
            # Location
            "Address": str(pii_input.get("Address","")).lower(),
            # Professional & Financial
            "Education Info": str(pii_input.get("Education Info","")).lower(),
            "Finance Status": str(pii_input.get("Finance Status","")).lower(),
            "Employer": str(pii_input.get("Employer","")).lower(),
            "Job Title": str(pii_input.get("Job Title","")).lower(),
            "Credit Score": str(pii_input.get("Credit Score","")).lower(),
        }
        allergies = str(pii_input.get("Allergies","None")).lower()
        if allergies != "none" and allergies: pii_values_to_check["Allergies"] = allergies
        disability = str(pii_input.get("Disability","None")).lower()
        if disability != "none" and disability: pii_values_to_check["Disability"] = disability

        text_for_phone_check = re.sub(r'\D', '', text_lower)
        text_for_alphanum_id_check = re.sub(r'[^a-z0-9]', '', text_lower)

        for field, pii_value_to_search in pii_values_to_check.items():
            original_pii_value = str(pii_input.get(field, "")) # Get original for debug
            
            # Skip if pii_value_to_search is empty
            if not pii_value_to_search:
                continue

            # Adjusted length check: allow short purely numeric strings, require len >=3 for others.
            # For address, we check parts, so main value length less critical here.
            if not (field != "Address" and not ((pii_value_to_search.isdigit() and len(pii_value_to_search) >= 1) or \
                                 (not pii_value_to_search.isdigit() and len(pii_value_to_search) >= 3)) ):
                pass # Continue to matching logic if length criteria met or it's an address
            elif field != "Address": # if length criteria not met and not Address, skip
                continue


            matched_this_field = False
            debug_value_that_matched = ""

            if field in ["Phone Number", "Work Phone", "Emergency Contact Phone"]:
                if pii_value_to_search and pii_value_to_search in text_for_phone_check:
                    matched_this_field = True
                    debug_value_that_matched = pii_value_to_search
            elif field in ["National ID", "Passport Number", "Driver's License"]:
                if pii_value_to_search and pii_value_to_search in text_for_alphanum_id_check:
                    matched_this_field = True
                    debug_value_that_matched = pii_value_to_search
            elif field == "Address":
                # Address logic: check for significant parts
                # Remove common, non-descriptive terms before splitting into parts
                address_cleaned_for_parts = pii_value_to_search
                for term in ["mount", "road", "street", "st", "rd", "ave", "avenue", "ln", "lane", "dr", "drive", "blvd", "boulevard", "ct", "court", "pl", "place", "pkwy", "parkway", "cir", "circle"]:
                    address_cleaned_for_parts = re.sub(r'\b' + re.escape(term) + r'\b', '', address_cleaned_for_parts)
                
                addr_parts = [part.strip() for part in address_cleaned_for_parts.split(',') if len(part.strip()) >= 3]
                addr_parts.extend([part.strip() for part in address_cleaned_for_parts.split() if len(part.strip()) >= 3])
                unique_addr_parts = sorted(list(set(p for p in addr_parts if p)), key=len, reverse=True) # Longer parts first

                for part in unique_addr_parts:
                    if part and part in text_lower:
                        matched_this_field = True
                        debug_value_that_matched = part # Log the specific part that matched
                        break 
            elif pii_value_to_search in text_lower: # General case
                matched_this_field = True
                debug_value_that_matched = pii_value_to_search
            
            if matched_this_field:
                mentioned_pii_fields_for_stats.append(field)
                row["debug_pii_mentions_details"].append({
                    "pii_unique_id": pii_input.get("Unique ID"),
                    "content_type": current_content_type,
                    "pii_field_name": field,
                    "original_pii_value_from_input": original_pii_value,
                    "pii_value_checked": debug_value_that_matched,
                    "synthetic_text": current_synthetic_text
                })

        # Social Media Handles
        social_media_handles = pii_input.get("Social Media Handles", {})
        if isinstance(social_media_handles, dict):
            for platform, handle in social_media_handles.items():
                normalized_handle = str(handle).lower()
                if normalized_handle and normalized_handle in text_lower:
                    mentioned_pii_fields_for_stats.append("Social Media Handle")
                    row["debug_pii_mentions_details"].append({
                        "pii_unique_id": pii_input.get("Unique ID"),
                        "content_type": current_content_type,
                        "pii_field_name": "Social Media Handle",
                        "original_pii_value_from_input": f"{platform}: {handle}",
                        "pii_value_checked": normalized_handle,
                        "synthetic_text": current_synthetic_text
                    })

        row["mentioned_pii_fields"] = list(set(mentioned_pii_fields_for_stats))
        row["num_mentioned_pii_fields"] = len(row["mentioned_pii_fields"])
        row["any_pii_mentioned"] = row["num_mentioned_pii_fields"] > 0
        extracted_rows.append(row)
    return extracted_rows

# --- 3. Plotting Utilities ---

def create_plot_dir(dir_name=OUTPUT_PLOT_DIR):
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    return dir_name

def sanitize_filename(name_part):
    return re.sub(r'[^\w\-_.]', '_', name_part)

def plot_distribution(data_series, title, xlabel, ylabel="Frequency", kind='bar', top_n=None, output_dir=OUTPUT_PLOT_DIR, palette=None, rotate_labels=False):
    plt.figure(figsize=(12, 7))
    current_palette = palette if palette is not None else categorical_palette
    safe_title_filename = sanitize_filename(title)

    if kind == 'hist':
        hist_palette_name = palette if palette else sequential_palette_name
        if isinstance(hist_palette_name, list):
             sns.histplot(data_series.dropna(), kde=True, color=hist_palette_name[0])
        else:
             sns.histplot(data_series.dropna(), kde=True, palette=hist_palette_name)
    elif kind == 'bar':
        counts = data_series.value_counts()
        if top_n: counts = counts.head(top_n)
        bar_palette = current_palette
        if isinstance(current_palette, list) and len(counts) > 0:
            bar_palette = [current_palette[i % len(current_palette)] for i in range(len(counts))] if len(counts) > len(current_palette) else current_palette[:len(counts)]
        elif not isinstance(current_palette, list) and len(counts) > 0:
            bar_palette = sns.color_palette(current_palette, len(counts))

        if not counts.empty:
            sns.barplot(x=counts.index, y=counts.values, palette=bar_palette)
            if rotate_labels or (len(counts.index) > 5 and counts.index.astype(str).str.len().max() > 10):
                 plt.xticks(rotation=45, ha="right")
        else: plt.text(0.5, 0.5, "No data to display", ha='center', va='center')
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_title_filename}.png"), dpi=300)
    plt.close()

def plot_stacked_bar(df_grouped, title, xlabel, ylabel, output_dir=OUTPUT_PLOT_DIR, palette=None):
    current_palette = palette if palette is not None else categorical_palette
    safe_title_filename = sanitize_filename(title)
    if df_grouped.empty:
        print(f"Skipping stacked bar plot '{title}' as data is empty.")
        plt.close(); return
    num_bars_per_group = len(df_grouped.columns)
    bar_palette = current_palette
    if isinstance(current_palette, list):
        bar_palette = [current_palette[i % len(current_palette)] for i in range(num_bars_per_group)] if num_bars_per_group > len(current_palette) else current_palette[:num_bars_per_group]
    else: bar_palette = sns.color_palette(current_palette, num_bars_per_group)

    df_grouped.plot(kind='bar', stacked=True, figsize=(14, 8), color=bar_palette)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=df_grouped.columns.name if hasattr(df_grouped.columns, 'name') else 'Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, f"{safe_title_filename}_stacked.png"), dpi=300)
    plt.close()

def plot_boolean_comparison(df, bool_column_name, compare_column_name, title, output_dir=OUTPUT_PLOT_DIR, top_n=None, palette=None):
    current_palette = palette if palette is not None else categorical_palette
    safe_title_filename = sanitize_filename(title)
    plt.figure(figsize=(14, 8))
    counts = df.groupby(bool_column_name)[compare_column_name].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    if counts.empty: print(f"Skipping boolean comparison plot '{title}' as no data after grouping."); plt.close(); return
    if top_n:
        top_categories = df[compare_column_name].value_counts().nlargest(top_n).index
        counts = counts[counts[compare_column_name].isin(top_categories)]
    if counts.empty: print(f"Skipping boolean comparison plot '{title}' as no data after top_n filtering."); plt.close(); return
    
    hue_palette = sns.color_palette(current_palette, 2) if not isinstance(current_palette, list) else current_palette[:2]
    sns.barplot(x=compare_column_name, y='percentage', hue=bool_column_name, data=counts, palette=hue_palette)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(compare_column_name, fontsize=12); plt.ylabel('Percentage within Group (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right"); plt.legend(title=bool_column_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_title_filename}.png"), dpi=300)
    plt.close()

def plot_pii_field_by_content_type(df_exploded_pii, title, output_dir=OUTPUT_PLOT_DIR, palette=None):
    current_palette = palette if palette is not None else categorical_palette
    safe_title_filename = sanitize_filename(title)
    if df_exploded_pii.empty or 'mentioned_pii_fields' not in df_exploded_pii.columns or 'content_type' not in df_exploded_pii.columns:
        print("Not enough data to plot PII field by content type."); return
    pii_content_counts = df_exploded_pii.groupby(['mentioned_pii_fields', 'content_type']).size().unstack(fill_value=0)
    if pii_content_counts.empty: print("No PII mentions found to plot by content type."); return
    num_pii_fields = len(pii_content_counts.index)
    num_content_types = len(pii_content_counts.columns)
    if num_content_types == 0: print("No content types found for PII field breakdown plot."); return

    bar_palette = current_palette
    if isinstance(current_palette, list):
        bar_palette = [current_palette[i % len(current_palette)] for i in range(num_content_types)] if num_content_types > len(current_palette) else current_palette[:num_content_types]
    else: bar_palette = sns.color_palette(current_palette, num_content_types)
    
    fig_width = max(15, num_pii_fields * 0.8 + num_content_types * 0.3) # Further tweaked for balance
    fig_height = 9 + num_pii_fields * 0.1 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    pii_content_counts.plot(kind='bar', ax=ax, width=0.75, color=bar_palette)
    
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("PII Field Mentioned", fontsize=14)
    ax.set_ylabel("Number of Mentions", fontsize=14)
    ax.tick_params(axis='x', labelsize=10, rotation=55, ha="right") # rotation adjusted
    ax.tick_params(axis='y', labelsize=10)
    
    # Legend below the chart
    # Determine optimal ncol: aim for 2 rows if many items, else 1 row
    legend_ncol = min(num_content_types, 4 if num_content_types > 8 else 3 if num_content_types > 4 else num_content_types) # Adaptive ncol
    legend_y_offset = -0.25 # Initial offset, adjust based on x-label length
    if num_pii_fields > 10 : legend_y_offset -= (num_pii_fields -10) * 0.005 # Lower legend more if many x-labels

    ax.legend(title="Content Type", loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), 
              ncol=legend_ncol, fancybox=True, shadow=False, fontsize='small', frameon=True, edgecolor='gray')
              
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.subplots_adjust(bottom=0.2 + abs(legend_y_offset*0.3) ) # Make space for legend & x-labels
    plt.savefig(os.path.join(output_dir, f"{safe_title_filename}.png"), dpi=300)
    plt.close(fig)

# --- 4. Main Analysis and Plotting Logic ---
def main():
    print(f"Loading data from {JSONL_FILE_PATH}...")
    raw_records = load_jsonl_data(JSONL_FILE_PATH)
    if not raw_records: print("No records loaded. Exiting."); return
    print(f"Loaded {len(raw_records)} raw records.")

    plot_dir = create_plot_dir()
    print(f"Plots and debug logs will be saved to: {plot_dir}")

    all_extracted_data = []
    for record in raw_records:
        all_extracted_data.extend(extract_features(record))

    df = pd.DataFrame(all_extracted_data)
    print(f"Created DataFrame with {len(df)} rows (one per content_pair) and {len(df.columns)} columns.")
    if df.empty: print("DataFrame is empty after feature extraction. Exiting."); return

    # --- Write Debug Information for PII Mentions ---
    print("\n--- Writing PII Mention Debug Information ---")
    all_debug_mentions_for_files = []
    if "debug_pii_mentions_details" in df.columns:
        for idx, row_data in df.iterrows():
            if row_data["debug_pii_mentions_details"]: # Check if the list is not empty
                all_debug_mentions_for_files.extend(row_data["debug_pii_mentions_details"])
    
    if all_debug_mentions_for_files:
        debug_df = pd.DataFrame(all_debug_mentions_for_files)
        if not debug_df.empty:
            for content_type_val, group in debug_df.groupby("content_type"):
                safe_content_type_name = sanitize_filename(content_type_val)
                debug_filename = os.path.join(plot_dir, f"debug_pii_mentions_for_{safe_content_type_name}.tsv")
                try:
                    group.to_csv(debug_filename, sep='\t', index=False,
                                 columns=["pii_unique_id", "content_type", "pii_field_name", "original_pii_value_from_input", "pii_value_checked", "synthetic_text"])
                    print(f"Saved debug info to {debug_filename}")
                except Exception as e:
                    print(f"Error writing debug file {debug_filename}: {e}")
        else:
            print("No detailed PII mentions found to write to debug files.")
    else:
        print("No 'debug_pii_mentions_details' found or list is empty; skipping debug file writing.")


    print("\n--- Basic Statistics ---")
    # ... (Basic stats print statements as before)
    print(f"Total number of synthetic content pieces: {len(df)}")
    unique_pii_profiles_count = df['pii_unique_id'].nunique()
    print(f"Number of unique PII profiles processed: {unique_pii_profiles_count}")
    if unique_pii_profiles_count > 0:
        print(f"Average synthetic texts per PII profile: {len(df) / unique_pii_profiles_count:.2f}")
    if 'generation_error' in df.columns:
        print(f"Number of records with generation errors: {df['generation_error'].sum()}")

    pii_df = df.drop_duplicates(subset=['pii_unique_id']).reset_index(drop=True) # For PII profile specific stats
    print(f"Unique PII profiles for PII-level analysis: {len(pii_df)}")

    # --- PII Input Analysis ---
    print("\n--- Analyzing PII Input Characteristics ---")
    if not pii_df.empty:
        for col in ['pii_age', 'pii_gender', 'pii_nationality', 'pii_job_title', 'pii_education_info', 'pii_locale', 'pii_marital_status', 'pii_finance_status', 'pii_net_worth', 'pii_annual_salary']:
            if col in pii_df.columns and pii_df[col].notna().any():
                kind = 'hist' if col in ['pii_age', 'pii_net_worth', 'pii_annual_salary'] else 'bar'
                palette_idx = ['pii_age', 'pii_net_worth', 'pii_annual_salary'].index(col) if col in ['pii_age', 'pii_net_worth', 'pii_annual_salary'] else None
                plot_distribution(pii_df[col].dropna(), f"{col.replace('pii_', '').replace('_', ' ').capitalize()} Distribution", col.replace('pii_', '').replace('_', ' ').capitalize(),
                                  kind=kind, output_dir=plot_dir, rotate_labels=True if kind=='bar' else False,
                                  palette=[categorical_palette[palette_idx % len(categorical_palette)]] if palette_idx is not None and kind == 'hist' else (categorical_palette if kind=='bar' else sequential_palette_name))
            else: print(f"Skipping plot for PII input '{col}' due to missing data.")
            
    # --- Synthetic Content Analysis ---
    print("\n--- Analyzing Synthetic Content Characteristics ---")
    if 'content_type' in df.columns and df['content_type'].notna().any():
        plot_distribution(df['content_type'].dropna(), "Distribution of Synthetic Content Types", "Content Type", output_dir=plot_dir, rotate_labels=True)
    if 'synthetic_text_length' in df.columns and df['synthetic_text_length'].notna().any():
        plot_distribution(df['synthetic_text_length'].dropna(), "Distribution of Synthetic Text Lengths", "Text Length (chars)", kind='hist', output_dir=plot_dir, palette=[categorical_palette[3 % len(categorical_palette)]])
    if 'synthetic_word_count' in df.columns and df['synthetic_word_count'].notna().any():
        plot_distribution(df['synthetic_word_count'].dropna(), "Distribution of Synthetic Word Counts", "Word Count", kind='hist', output_dir=plot_dir, palette=[categorical_palette[4 % len(categorical_palette)]])

    # --- PII Memorization/Mention Analysis ---
    print("\n--- Analyzing PII Mentions in Synthetic Content ---")
    if 'any_pii_mentioned' in df.columns:
        print(f"Total synthetic texts with at least one PII mention: {df['any_pii_mentioned'].sum()} ({df['any_pii_mentioned'].mean()*100:.2f}%)")
    if 'num_mentioned_pii_fields' in df.columns and df['num_mentioned_pii_fields'].notna().any():
        plot_distribution(df['num_mentioned_pii_fields'].dropna(), "Distribution of Num PII Fields Mentioned per Text", "Number of PII Fields", kind='hist', output_dir=plot_dir, palette=[categorical_palette[5 % len(categorical_palette)]])

    all_mentioned_fields_list = [field for sublist in df['mentioned_pii_fields'] for field in sublist if field] # Ensure field is not None
    if all_mentioned_fields_list:
        mentioned_field_counts = Counter(all_mentioned_fields_list)
        mf_series = pd.Series(mentioned_field_counts).sort_values(ascending=False)
        if not mf_series.empty:
            plot_distribution(mf_series, "Most Frequently Mentioned PII Fields (Top N)", "PII Field", "Frequency", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)
    else: print("No PII fields were detected as mentioned across all texts.")

    # --- Breakdown Chart: PII Field Mentions by Content Type ---
    print("\n--- Generating PII Field Mentions by Content Type Chart ---")
    if 'mentioned_pii_fields' in df.columns and df['any_pii_mentioned'].sum() > 0 :
        # Use the 'mentioned_pii_fields' which contains unique field names per text for this plot's aggregation
        df_exploded_for_plot = df[df['any_pii_mentioned']][['mentioned_pii_fields', 'content_type']].explode('mentioned_pii_fields')
        df_exploded_for_plot.dropna(subset=['mentioned_pii_fields'], inplace=True)
        if not df_exploded_for_plot.empty:
             plot_pii_field_by_content_type(df_exploded_for_plot,
                                       "Breakdown of PII Field Mentions by Content Type",
                                       output_dir=plot_dir)
        else: print("No PII mentions to analyze for the breakdown chart after exploding and NaN removal.")
    else: print("Skipping PII Field Mentions by Content Type chart: No PII mentions found or relevant columns missing.")

    # --- Cross-Analysis ---
    print("\n--- Cross-Analysis: PII Input vs. Output ---")
    # (Simplified calls to plotting functions, assuming they handle missing data checks internally)
    if 'any_pii_mentioned' in df.columns and 'content_type' in df.columns:
        pii_mention_by_content_type = df.groupby('content_type')['any_pii_mentioned'].mean().mul(100).sort_values(ascending=False)
        if not pii_mention_by_content_type.empty:
            plt.figure(figsize=(12, 7))
            current_pal = categorical_palette[:len(pii_mention_by_content_type)] if isinstance(categorical_palette, list) else categorical_palette
            sns.barplot(x=pii_mention_by_content_type.index, y=pii_mention_by_content_type.values, palette=current_pal)
            plt.title("Percentage of Texts with PII Mentions by Content Type", fontsize=16, pad=20)
            plt.xlabel("Content Type", fontsize=12); plt.ylabel("Texts with PII Mentions (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right"); plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, sanitize_filename("pii_mentions_by_content_type")+".png"), dpi=300)
            plt.close()

    if 'pii_age' in df.columns and 'num_mentioned_pii_fields' in df.columns and df['pii_age'].notna().any():
        max_age_val = df['pii_age'].max() if df['pii_age'].notna().any() else 75.0
        bins = [0, 18, 30, 45, 60, 75, max(max_age_val + 1.0, 76.0)]
        labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '75+']
        final_bins = [b for b in bins if b <= max_age_val + 1.0]
        if not final_bins or (len(final_bins) == 1 and final_bins[0] == 0): final_bins = [0, max(max_age_val + 1.0, 1.0)]
        if final_bins[-1] < max_age_val + 1 and len(final_bins) < len(bins): final_bins[-1] = max_age_val + 1.0
        final_labels = labels[:len(final_bins)-1]

        if final_bins and final_labels and len(final_bins) == len(final_labels) + 1:
            df['pii_age_group'] = pd.cut(df['pii_age'], bins=final_bins, labels=final_labels, right=False, include_lowest=True)
            avg_mentions_by_age = df.groupby('pii_age_group', observed=True)['num_mentioned_pii_fields'].mean().sort_values(ascending=False)
            if not avg_mentions_by_age.empty:
                plt.figure(figsize=(12, 7))
                current_pal = categorical_palette[:len(avg_mentions_by_age)] if isinstance(categorical_palette, list) else categorical_palette
                sns.barplot(x=avg_mentions_by_age.index, y=avg_mentions_by_age.values, palette=current_pal)
                plt.title("Average Number of PII Fields Mentioned by PII Age Group", fontsize=16, pad=20)
                plt.xlabel("PII Age Group", fontsize=12); plt.ylabel("Avg. Num PII Fields Mentioned", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, sanitize_filename("avg_pii_mentions_by_age_group")+".png"), dpi=300)
                plt.close()
        else: print(f"Skipping PII age group plot due to binning issues. Bins: {final_bins}, Labels: {final_labels}")

    if 'any_pii_mentioned' in df.columns and 'content_type' in df.columns:
        plot_boolean_comparison(df, 'any_pii_mentioned', 'content_type',
                                "Content Type Dist - PII Mentioned vs. Not Mentioned",
                                output_dir=plot_dir, top_n=TOP_N_CATEGORIES)

    if 'pii_job_title' in df.columns and 'any_pii_mentioned' in df.columns and df['pii_job_title'].notna().any():
        top_job_titles = df['pii_job_title'].value_counts().nlargest(TOP_N_CATEGORIES).index
        df_top_jobs = df[df['pii_job_title'].isin(top_job_titles)]
        if not df_top_jobs.empty:
            mentions_by_job = df_top_jobs.groupby('pii_job_title')['any_pii_mentioned'].mean().mul(100).sort_values(ascending=False)
            if not mentions_by_job.empty:
                plt.figure(figsize=(14, 8))
                current_pal = categorical_palette[:len(mentions_by_job)] if isinstance(categorical_palette, list) else categorical_palette
                sns.barplot(x=mentions_by_job.index, y=mentions_by_job.values, palette=current_pal)
                plt.title(f"PII Mention Rate by PII Job Title (Top {len(mentions_by_job)})", fontsize=16, pad=20)
                plt.xlabel("PII Job Title", fontsize=12); plt.ylabel("Texts with PII Mentions (%)", fontsize=12)
                plt.xticks(rotation=45, ha="right"); plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, sanitize_filename(f"pii_mentions_by_top_{len(mentions_by_job)}_job_titles")+".png"), dpi=300)
                plt.close()

    if 'pii_job_title' in df.columns and 'content_type' in df.columns and df['pii_job_title'].notna().any() and df['content_type'].notna().any():
        num_top_jobs_stacked = 8
        top_jobs_for_stacked = df['pii_job_title'].value_counts().nlargest(num_top_jobs_stacked).index
        df_top_jobs_stacked = df[df['pii_job_title'].isin(top_jobs_for_stacked)]
        if not df_top_jobs_stacked.empty:
            job_content_type_dist = df_top_jobs_stacked.groupby('pii_job_title')['content_type'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
            if not job_content_type_dist.empty:
                plot_stacked_bar(job_content_type_dist,
                                 f"Content Type Distribution per PII Job Title (Top {len(job_content_type_dist)})",
                                 "PII Job Title", "Percentage of Content Types (%)", output_dir=plot_dir)

    if 'generation_error' in df.columns and df['generation_error'].any():
        print("\n--- Analyzing Generation Errors ---")
        error_pii_profiles = df[df['generation_error']].drop_duplicates(subset=['pii_unique_id'])
        print(f"Total unique PII profiles associated with generation errors: {len(error_pii_profiles)}")
        if not error_pii_profiles.empty and error_pii_profiles['pii_job_title'].notna().any():
            plot_distribution(error_pii_profiles['pii_job_title'].dropna(), "Job Titles in PII Profiles with Errors (Top N)", "Job Title", top_n=TOP_N_CATEGORIES, output_dir=plot_dir, rotate_labels=True)

    print(f"\nAnalysis complete. All plots and debug logs saved to '{plot_dir}'.")

if __name__ == "__main__":
    if not os.path.exists(JSONL_FILE_PATH):
        print(f"ERROR: The data file was not found at '{JSONL_FILE_PATH}'.")
        print("Please ensure the JSONL_FILE_PATH variable in the script is set correctly.")
    else:
        main()