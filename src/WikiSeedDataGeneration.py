import itertools
import re
import csv
import logging
import pandas as pd
from datasets import load_dataset
from datetime import datetime
import json
from ethnicolr import pred_wiki_ln, pred_census_ln
import warnings

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_processing.log"),
        logging.StreamHandler()
    ]
)

# --- Helper Functions ---
def safe_serialize(text):
    """Ensure text is properly serialized to escape newlines and tabs."""
    if isinstance(text, str):
        return json.dumps(text, ensure_ascii=False)  # Serialize like C#'s SerializeObject
    return text  # If not a string, return as-is

def safe_deserialize(serialized_text):
    """Deserialize text while handling escaped newlines and tabs."""
    if isinstance(serialized_text, str):
        try:
            return json.loads(serialized_text)  # Deserialize like C#'s DeserializeObject
        except json.JSONDecodeError:
            return serialized_text  # Return as-is if deserialization fails
    return serialized_text  # If not a string, return as-is

def extract_section(sections, target_section):
    """
    Recursively searches for a section by name and extracts its text content.

    Args:
        sections (list): List of section objects from the Wikipedia dataset.
        target_section (str): The name of the section to extract.

    Returns:
        str: Extracted text from the section (including sub-sections).
    """
    def recursive_extract(section_list):
        extracted_text = []
        
        if not section_list or not isinstance(section_list, list):
            logging.debug("Invalid sections format. Expected a list.")
            return extracted_text
        
        for section in section_list:
            if str(target_section).lower() in str(section.get("name", "")).strip().lower():
                extracted_text.extend(collect_text_from_section(section))
            if "has_parts" in section:
                extracted_text.extend(recursive_extract(section["has_parts"]))
        return extracted_text

    def collect_text_from_section(section):
        """Collects text from paragraphs within a section."""
        texts = []
        if "has_parts" in section:
            if section["has_parts"] is None or not isinstance(section["has_parts"], list):
                logging.debug(f"Invalid has_parts format in section: {section.get('name', 'Unknown')}")
                return texts
            
            for part in section["has_parts"]:
                if part.get("type") == "paragraph" and "value" in part:
                    texts.append(part["value"])
        return texts

    extracted_text = recursive_extract(sections)
    result_text = "\n".join(extracted_text).strip()
    
    logging.debug(f"Extracted '{target_section}' section: {result_text[:100]}...")  # Log first 100 chars
    return result_text

def extract_origin_from_infobox(infoboxes):
    for info in infoboxes:
        if str(info.get("name", "")).lower() == "infobox person":
            if isinstance(info.get("has_parts", []), list):
                for part in info["has_parts"]:
                    if isinstance(part, dict) and part.get("name", "").lower() == "born":
                        logging.debug(f"Extracted origin from infobox: {part.get('value', '')}")
                        return part.get("value", "")
    return ""

def get_ethnicity(name, conf_int=0.9):
    """
    Function to predict the ethnicity of a given name using pred_wiki_ln and pred_census_ln.
    
    Parameters:
        name (str): Full name (e.g., "Jothi Patel").
        conf_int (float): Confidence interval for prediction (default is 0.9).
        
    Returns:
        dict: Dictionary containing ethnicity predictions from both pred_wiki_ln and pred_census_ln.
    """
    # Extract the last name
    parts = name.split()
    last_name = parts[-1] if parts else None  # Handle edge case if name is empty or None
    
    if last_name:
        # Create a DataFrame with the last name
        df = pd.DataFrame({'last': [last_name]})
        
        # Run predictions using pred_wiki_ln and pred_census_ln
        wiki_result = pred_wiki_ln(df, 'last', conf_int=conf_int)
        census_result = pred_census_ln(df, 'last', conf_int=conf_int)
        
        # Get ethnicity from both prediction results
        wiki_ethnicity = wiki_result.iloc[0].get('race') if not wiki_result.empty else None
        census_ethnicity = census_result.iloc[0].get('race') if not census_result.empty else None
        
        # Return both results as a dictionary
        return {
            'wiki_ethnicity': wiki_ethnicity,
            'census_ethnicity': census_ethnicity
        }
    else:
        return {
            'wiki_ethnicity': None,
            'census_ethnicity': None
        }

# --- Main Processing Function ---

def process_dataset(limit=10000):
    logging.info("Starting dataset processing...")

    dataset = load_dataset("wikimedia/structured-wikipedia", "20240916.en", split="train", streaming=True)
    
    # Skip the first 1003459 records in dataset
    dataset = itertools.islice(dataset, 1003459, None)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/wikipedia_people_{timestamp}.tsv"
    count = 0
    index = 0

    with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
        fieldnames = [
            "Name", "URL", "Abstract", 
            "Wikipedia_Content", "Personal_Life", "Early_Life"
        ]
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for record in dataset:
            index += 1
            #logging.info(f"Processing record {index}: {record.get('name', 'Unknown')}")
            infoboxes = record.get("infoboxes", [])
            if infoboxes is None or not isinstance(infoboxes, list):
                logging.debug(f"Skipping record due to missing or invalid infoboxes: {record.get('name', 'Unknown')}")
                continue
            
            is_person = any(str(infobox.get("name", "")).lower() == "infobox person" for infobox in infoboxes)
            if not is_person:
                logging.debug(f"Skipping non-person record: {record.get('name', 'Unknown')}")
                continue
            
            name = record.get("name", "")
            if not name:
                logging.warning("Skipping record without a valid name.")
                continue  # Skip if no name is found
            
            abstract = record.get("abstract", "")
            url = record.get("url", "")

            #origin = extract_origin_from_infobox(infoboxes)
            
            # Extract sections properly using the new method
            sections = record.get("sections", [])
            full_text = sections
            if not sections or not isinstance(sections, list):
                logging.debug(f"Skipping record due to missing or invalid sections: {record.get('name', 'Unknown')}")
                personal_life = ""
                early_life = ""
            else:
                personal_life = extract_section(sections, "Personal life")
                early_life = extract_section(sections, "Early life")

            # If personal_life and early_life are empty, skip the record
            if not personal_life and not early_life:
                logging.debug(f"Skipping record with no personal or early life sections: {record.get('name', 'Unknown')}")
                continue
            
            # ethnicity_results = get_ethnicity(name)
            # wiki_race = ethnicity_results["wiki_ethnicity"]
            # census_race = ethnicity_results["census_ethnicity"]
            
            logging.info(f"Total processed {index} successful {count} name {record.get('name', 'Unknown')}")
            writer.writerow({
                "Name": safe_serialize(name),
                "URL": safe_serialize(url),
                "Abstract": safe_serialize(abstract),
                "Wikipedia_Content": safe_serialize(full_text),
                "Personal_Life": safe_serialize(personal_life),
                "Early_Life": safe_serialize(early_life)
            })

            count += 1
            logging.info(f"Processed record {count}: {name}")
            if count >= limit:
                break

    logging.info(f"Finished processing {count} person records into {output_file}")

# --- Execution ---

if __name__ == "__main__":
    process_dataset(limit=50000)