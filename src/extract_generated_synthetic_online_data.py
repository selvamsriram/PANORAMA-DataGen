import json
import re
from typing import List, Tuple, Dict
from collections import defaultdict
import logging
from datasets import Dataset, DatasetInfo, Features, Value
from typing import List, Dict, Union, Optional
import ast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

class ContentProcessingError(Exception):
    """Custom exception for content processing errors"""
    pass

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

def extract_content_pairs(text: str, synthetic_pii_input: dict) -> List[Tuple[str, str]]:
    """Extract content type and text pairs from the model output."""
    
    if not isinstance(synthetic_pii_input, dict):
        raise ContentProcessingError(f"Expected dict input, got {type(synthetic_pii_input)}")
    
    first_name = synthetic_pii_input.get('First Name', None)
    social_media_handles = synthetic_pii_input.get('Social Media Handles', None)
    social_media_handles = parse_social_handles(social_media_handles, context="handle_mentioned")
    
    available_handles = []

    for v in social_media_handles.values():
        if isinstance(v, list):
            available_handles.extend(v)
        else:
            available_handles.append(v)
            logging.debug(f"Available handles: {available_handles} for record")

    available_handles.append(first_name)
    
    if not isinstance(text, str):
        raise ContentProcessingError(f"Expected string input, got {type(text)}")
    
    # Define the content types and their variations
    content_types = {
        'Social Media': [
            r'\[Social Media\]',
            r'\[Social\]',
            r'\[Social Post\]',
            r'\[Social Media Post\]',
            r'\[Social Network\]',
            r'\[Social Network Post\]'
        ],
        'Forum Post': [
            r'\[Forum Post\]',
            r'\[Forum\]',
            r'\[Forum Message\]',
            r'\[Forum Thread\]',
            r'\[Discussion Forum\]',
            r'\[Forum Discussion\]'
        ],
        'Online Review': [
            r'\[Online Review\]',
            r'\[Review\]',
            r'\[Product Review\]',
            r'\[Service Review\]',
            r'\[Customer Review\]',
            r'\[User Review\]'
        ],
        'Blog/News Article Comment': [
            r'\[Blog/News Article Comment\]',
            r'\[Blog Comment\]',
            r'\[News Comment\]',
            r'\[Article Comment\]',
            r'\[Comment\]',
            r'\[Reader Comment\]'
        ],
        'Online Ad': [
            r'\[Online Ad\]',
            r'\[Advertisement\]',
            r'\[Promotion\]',
            r'\[Marketing\]',
            r'\[Promotional Content\]'
        ]
    }
    
    content_pairs = []
    
    # Process each content type and its variations
    for base_type, variations in content_types.items():
        # Create a pattern that matches any of the variations
        pattern = '|'.join(variations)
        
        # Find all matches of this content type
        matches = re.finditer(f'({pattern})(.*?)(?=\[|$)', text, re.DOTALL)
        
        for match in matches:
            content = match.group(2).strip()
            if content:  # Only add if there's actual content
                if base_type != 'Social Media' and base_type != 'Forum Post':
                    content_pairs.append((base_type, content))
                else:
                    mentioned_handle = handle_mentioned(content, first_name, social_media_handles)
                    if mentioned_handle:
                        content_pairs.append((base_type, content))
                        logging.info(f"Handle mentioned in text: '{content}'.  Adding text as is.")
                    else:
                        prefix = f"{available_handles[0]} " if available_handles else ""
                        content_pairs.append((base_type, prefix + ": "+ content))
                        logging.info(f"No handle mentioned in text: '{content}'.  Adding prefix: '{prefix}'")
    
    if not content_pairs:
        logging.warning(f"No valid content pairs found in text: {text[:100]}...")
    
    return content_pairs

def process_record(record: dict) -> dict:
    """Process a single record and add content pairs to SyntheticTrainingData."""
    try:
        if not isinstance(record, dict):
            raise ContentProcessingError(f"Expected dict input, got {type(record)}")
        
        if 'synthetic_pii_input' not in record:
            logging.warning("Record missing synthetic_pii_input field")
            return record
        
        if 'SyntheticTrainingData' not in record:
            logging.warning("Record missing SyntheticTrainingData field")
            return record
            
        if 'model_output' not in record['SyntheticTrainingData']:
            logging.warning("Record missing model_output field")
            return record
            
        content_pairs = extract_content_pairs(record['SyntheticTrainingData']['model_output'], record['synthetic_pii_input'])
        record['SyntheticTrainingData']['content_pairs'] = [
            {'ContentType': pair[0], 'Text': pair[1]} 
            for pair in content_pairs
        ]
        return record
        
    except Exception as e:
        logging.error(f"Error processing record: {str(e)}")
        return record

def analyze_content_types(records: List[dict]) -> Dict[str, dict]:
    """Analyze content types and generate statistics."""
    content_type_stats = defaultdict(lambda: {'count': 0, 'total_length': 0})
    total_records = len(records)
    
    for record in records:
        if 'SyntheticTrainingData' in record and 'content_pairs' in record['SyntheticTrainingData']:
            for pair in record['SyntheticTrainingData']['content_pairs']:
                content_type = pair['ContentType']
                content_type_stats[content_type]['count'] += 1
                content_type_stats[content_type]['total_length'] += len(pair['Text'])
    
    # Calculate averages
    for content_type in content_type_stats:
        content_type_stats[content_type]['average_length'] = (
            content_type_stats[content_type]['total_length'] / 
            content_type_stats[content_type]['count']
        )
        content_type_stats[content_type]['percentage_of_records'] = (
            content_type_stats[content_type]['count'] / total_records * 100
        )
    
    return dict(content_type_stats)

def create_pretraining_data(record: dict) -> List[Dict[str, str]]:
    """Create pre-training data records from a single input record.
    
    Args:
        record: The input record containing synthetic data
        
    Returns:
        List of pre-training records with UniqueID and Text fields
    """
    pretraining_records = []
    
    try:
        # Add the synthetic passage as one record
        if 'SyntheticPassageGeneration' in record and 'generated_synthetic_passage' in record['SyntheticPassageGeneration']:
            passage = record['SyntheticPassageGeneration']['generated_synthetic_passage']
            if passage and isinstance(passage, str):
                pretraining_records.append({
                    'UniqueID': record['synthetic_pii_input']['Unique ID'],
                    'ContentType': "Article",
                    'Text': passage.strip()
                })
        
        # Add content pairs as separate records
        if 'SyntheticTrainingData' in record and 'content_pairs' in record['SyntheticTrainingData']:
            for pair in record['SyntheticTrainingData']['content_pairs']:
                if 'Text' in pair and pair['Text']:
                    pretraining_records.append({
                        'UniqueID': record['synthetic_pii_input']['Unique ID'],
                        'ContentType': pair['ContentType'],
                        'Text': pair['Text'].strip()
                    })
    
    except Exception as e:
        logging.error(f"Error creating pre-training data: {str(e)}")
    
    return pretraining_records

def main():
    input_file = '/Users/sriramselvam/Code/PANORAMA-DataGen/data/Azure_Synthetic_Data_10K.combined.jsonl'
    output_file = '/Users/sriramselvam/Code/PANORAMA-DataGen/data/Azure_Synthetic_Data_10K.processed.jsonl'
    pretraining_file = '/Users/sriramselvam/Code/PANORAMA-DataGen/data/tsv_variants/Azure_Synthetic_Data_10K.pretraining.tsv'
    pretraining_hf_file = '/Users/sriramselvam/Code/PANORAMA-DataGen/data/Azure_Synthetic_Data_10K.pretraining.hf.tsv'
    stats_file = 'content_type_stats.json'
    repo_id = "srirxml/PANORAMA"
    
    processed_records = []
    pretraining_records = []
    error_count = 0
    total_records = 0
    
    try:
        with open(input_file, 'r') as infile, \
             open(output_file, 'w') as outfile, \
             open(pretraining_file, 'w', encoding='utf-8') as pretraining_outfile:
            
            # Write TSV header
            pretraining_outfile.write("UniqueID\tContentType\tText\n")
            
            for line_num, line in enumerate(infile, 1):
                total_records += 1
                try:
                    record = json.loads(line.strip())
                    processed_record = process_record(record)
                    processed_records.append(processed_record)
                    outfile.write(json.dumps(processed_record) + '\n')
                    
                    # Create and write pre-training records
                    pretraining_data = create_pretraining_data(processed_record)
                    # write pre-training records to TSV
                    for pretraining_record in pretraining_data:
                        pretraining_outfile.write(f"{pretraining_record['UniqueID']}\t{pretraining_record.get('ContentType', None)}\t{json.dumps(pretraining_record['Text'])}\n")

                        pretraining_records.append({
                            "id": pretraining_record["UniqueID"],
                            "content-type": pretraining_record.get("ContentType", None),
                            "text": pretraining_record["Text"]
                        })
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    logging.error(f"JSON decode error on line {line_num}: {str(e)}")
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing line {line_num}: {str(e)}")
            
            
            # Define your schema explicitly
            features = Features({
                "id": Value("string"),           # Adjust keys/values to match your pretraining_records
                "content-type": Value("string"),          # Remove or modify if you don't have a label field
                "text": Value("string")           # Remove or modify if you don't have a label field
            })

            # Create dataset with metadata
            hf_dataset = Dataset.from_list(pretraining_records)

            # Save and push
            hf_dataset.save_to_disk(pretraining_hf_file)
            hf_dataset.push_to_hub(repo_id)
            
        # Generate and save statistics
        stats = analyze_content_types(processed_records)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Log summary statistics
        logging.info(f"\nProcessing Summary:")
        logging.info(f"Total records processed: {total_records}")
        logging.info(f"Records with errors: {error_count}")
        logging.info(f"Successfully processed records: {len(processed_records)}")
        logging.info(f"Pre-training records created: {len(pretraining_records)}")
        logging.info(f"Unique content types found: {len(stats)}")
        
        logging.info("\nContent Type Statistics:")
        for content_type, data in stats.items():
            logging.info(f"\n{content_type}:")
            logging.info(f"  Total occurrences: {data['count']}")
            logging.info(f"  Average length: {data['average_length']:.2f} characters")
            logging.info(f"  Percentage of records: {data['percentage_of_records']:.2f}%")
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()