from datasets import load_dataset, Dataset
import json
import logging
from typing import List, Dict, Union, Optional
import ast

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_social_handles(handles: Union[str, dict, None], context: str = "") -> Dict:
    """
    Safely parses social media handles from a string or dictionary.

    Args:
        handles: A stringified or actual dictionary of handles.
        context: Optional context string for logging.

    Returns:
        A dictionary of social media handles.
    """
    if not handles:
        return {}

    if isinstance(handles, dict):
        return handles

    if isinstance(handles, str):
        try:
            return json.loads(handles)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(handles)
            except (ValueError, SyntaxError):
                logging.error(f"[{context}] Could not parse social_media_handles: {handles}. Returning empty dict.")
                return {}

    logging.error(f"[{context}] Unexpected type for social_media_handles: {type(handles)}. Returning empty dict.")
    return {}

def handle_mentioned(text: str, first_name: str, handles: Union[dict, str, None]) -> Optional[str]:
    """Checks if any handle is mentioned in the given text (case-insensitive)."""
    handles = parse_social_handles(handles, context="handle_mentioned")
    handles['First Name'] = first_name  # Ensure first name is included in handles
    for handle_type, handle_value in handles.items():
        if handle_value:
            if isinstance(handle_value, list):
                for h in handle_value:
                    if h.lower() in text.lower():
                        return h
            elif isinstance(handle_value, str) and handle_value.lower() in text.lower():
                return handle_value
    return None

def process_panorama_record(record: dict) -> dict:
    unique_id = record['Unique ID']
    #logging.info(f"Processing record with Unique ID: {unique_id}")

    complete_info = record.get('complete_info', {})
    if isinstance(complete_info, str):
        try:
            complete_info = json.loads(complete_info)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON for 'complete_info' in record: {unique_id}.  Setting to empty dict.")
            complete_info = {}

    locale = record['Locale']
    age = record['Age']
    if not isinstance(age, (int, float)):
        try:
            age = int(age)
        except ValueError:
            logging.error(f"Invalid age value '{age}' for record: {unique_id}.  Setting age to 0.")
            age = 0

    first_name = record['First Name']
    last_name = record['Last Name']
    social_media_handles_raw = record['Social Media Handles']
    social_media_handles = parse_social_handles(social_media_handles_raw, context=f"process_panorama_record {unique_id}")
    content_pairs = complete_info.get('SyntheticTrainingData', {}).get('content_pairs', [])
    result_strings = []
    available_handles = []

    for v in social_media_handles.values():
        if isinstance(v, list):
            available_handles.extend(v)
        else:
            available_handles.append(v)
            logging.debug(f"Available handles: {available_handles} for record: {unique_id}")
            
    for pair in content_pairs:
        text = pair['Text']
        content_type = pair['ContentType']
            
        logging.debug (f"ContentType: {content_type}, Text: '{text}'")
        mentioned_handle = handle_mentioned(text, first_name, social_media_handles)
        if mentioned_handle:
                result_strings.append(text)
                logging.info(f"Handle mentioned in text: '{text}' for record: {unique_id}.  Adding text as is.")
        else:
                prefix = f"{available_handles[0]} " if available_handles else ""
                result_strings.append(prefix + text)
                logging.info(f"No handle mentioned in text: '{text}' for record: {unique_id}.  Adding prefix: '{prefix}'")
                
    return {
        "Unique ID": unique_id,
        "AllContent": result_strings
    }

def process_dataset(dataset_name: str = "srirxml/PANORAMA-Plus", split: str = "train") -> Dataset:
    dataset = load_dataset(dataset_name, split=split)
    logging.info(f"Loaded dataset: {dataset_name}, split: {split}")

    total_valid_records = 0
    processed_records = []
    for record in dataset:
        process_result = process_panorama_record(record)
        if process_result['AllContent']:
            processed_records.append(process_result)
            total_valid_records += 1

    logging.info(f"Total valid records: {total_valid_records}")
    return Dataset.from_list(processed_records)

def upload_dataset(dataset: Dataset, repo_name: str = "PANORAMA-IdCtText") -> None:
    dataset.push_to_hub(repo_name)
    logging.info(f"Uploaded dataset to https://huggingface.co/datasets/{repo_name}")
    print(f"Uploaded dataset to https://huggingface.co/datasets/{repo_name}")

if __name__ == "__main__":
    processed_dataset = process_dataset()
