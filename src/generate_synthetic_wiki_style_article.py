from datetime import datetime
import os
import json
import argparse
import logging
from openai import AzureOpenAI, OpenAIError
import uuid # For custom_id if needed

# --- Configuration ---
# Azure OpenAI endpoint details (can be overridden by arguments)
DEFAULT_ENDPOINT = "FILL_IN_HERE" # Use your actual Azure OpenAI endpoint
DEFAULT_API_VERSION = "2024-12-01-preview" # Use a specific, tested version
# Model deployment names
LIVE_MODEL_DEPLOYMENT = "o3-mini"
BATCH_MODEL_DEPLOYMENT = "o3-mini-batch" # As specified by user

# Recommended generation parameters (used in both modes for metadata/request body)
MAX_TOKENS = 2000

# --- Helper Functions ---

def setup_logging(log_level_str):
    """Configures logging based on the provided level string."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    # Basic configuration, consider adding file handler if needed
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.info(f"Logging level set to: {log_level_str.upper()}")

def read_api_key(filepath):
    """Reads the API key from the first line of a file."""
    logging.debug(f"Attempting to read API key from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            key = f.readline().strip()
            if not key:
                logging.error(f"API key file '{filepath}' is empty.")
                raise ValueError("API key file is empty.")
            logging.debug("Successfully read API key.")
            return key
    except FileNotFoundError:
        logging.error(f"Error: API key file not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading API key from {filepath}: {e}")
        raise

def read_prompt(filepath):
    """Reads the entire content of the prompt file."""
    logging.debug(f"Attempting to read prompt file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
            logging.debug(f"Successfully read prompt file (length: {len(prompt_content)}).")
            return prompt_content
    except FileNotFoundError:
        logging.error(f"Error: Prompt file not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading prompt file {filepath}: {e}")
        raise

def call_azure_openai(client, system_prompt, user_prompt, model_deployment, max_tokens):
    """
    Calls the Azure OpenAI Chat Completions API (Live Mode) and handles errors.
    Returns the content string or None on failure.
    """
    logging.info(f"Calling Azure OpenAI (Live) model: {model_deployment}")
    logging.debug(f"System Prompt Length: {len(system_prompt)}")
    logging.debug(f"User Prompt Length: {len(user_prompt)}")
    logging.debug(f"Parameters: max_tokens={max_tokens}")

    try:
        response = client.chat.completions.create(
            model=model_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max_tokens,
        )
        # Check if response and choices are valid before accessing
        if response and response.choices and len(response.choices) > 0:
             model_output = response.choices[0].message.content
             logging.info("Successfully received response from Azure OpenAI.")
             logging.debug(f"Response length: {len(model_output) if model_output else 0}")
             return model_output
        else:
             logging.warning("Received an empty or invalid response from Azure OpenAI.")
             return None

    except OpenAIError as api_err:
        logging.error(f"Azure OpenAI API Error: {api_err}")
        return None # Indicate failure
    except Exception as call_err:
         logging.error(f"Unexpected error during API call: {call_err}", exc_info=True)
         return None # Indicate failure

# --- Pipeline Functions ---

def process_single_record_live(line_num, synthetic_record, real_record, system_prompt_content, client):
    """
    Processes a single record for LIVE inferencing.
    Returns a dictionary containing inputs, parameters, and output/error.
    """
    logging.debug(f"Processing record (Live): {line_num}")
    # Initialize output structure
    output_data = {
        "line_number": line_num,
        "synthetic_pii_input": synthetic_record,
        "real_person_text_input": None,
        "system_prompt": system_prompt_content,
        "user_prompt": None,
        "model_parameters": {
            "model": LIVE_MODEL_DEPLOYMENT, # Use live model name
            "max_completion_tokens": MAX_TOKENS,
        },
        "model_output": None,
        "error": None
    }

    try:
        # Extract required text from the real profile
        abstract = real_record.get('Abstract', '') or ''
        personal_life = real_record.get('Personal_Life', '') or ''
        early_life = real_record.get('Early_Life', '') or ''
        real_person_text = f"{abstract}\n{personal_life}\n{early_life}".strip()
        if not real_person_text:
             logging.warning(f"Real person text is empty for line {line_num}. Proceeding with empty text.")
        output_data["real_person_text_input"] = real_person_text
        logging.debug(f"Extracted real person text (length: {len(real_person_text)}) for line {line_num}.")

        # Format synthetic PII details as a JSON string for the prompt
        synthetic_pii_details_str = json.dumps(synthetic_record, indent=2)

        # Construct the User Prompt
        user_prompt_content = f"""**Inputs**:\nreal_wiki_inspiration_text:\n{real_person_text} \n synthetic_profile_json: \n {synthetic_pii_details_str}"""
        output_data["user_prompt"] = user_prompt_content

        # Call the modular API function
        model_output = call_azure_openai(
            client=client,
            system_prompt=system_prompt_content,
            user_prompt=user_prompt_content,
            model_deployment=LIVE_MODEL_DEPLOYMENT, # Use live model
            max_tokens=MAX_TOKENS
        )

        if model_output is not None:
            output_data["model_output"] = model_output
            logging.info(f"Successfully processed record {line_num} (Live).")
        else:
            output_data["error"] = "API call failed or returned empty response. See logs."
            logging.warning(f"API call failed for record {line_num} (Live).")

    except KeyError as ke:
         logging.error(f"Missing expected key in input record at line {line_num}: {ke}")
         output_data["error"] = f"Data processing error: Missing key '{ke}'"
    except Exception as proc_err:
        logging.error(f"Error processing data for record {line_num} (Live): {proc_err}", exc_info=True)
        output_data["error"] = f"Data processing error: {proc_err}"

    return output_data


def run_live_pipeline(api_key_file, prompt_file, synthetic_file, real_file, output_file, azure_endpoint, api_version):
    """
    Pipeline function for LIVE inferencing.
    """
    logging.info("--- Running in LIVE Inference Mode ---")
    try:
        # --- Initialization ---
        api_key = read_api_key(api_key_file)
        system_prompt_content = read_prompt(prompt_file)

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
        )
        logging.info(f"Initialized AzureOpenAI client for LIVE mode. Endpoint: {azure_endpoint}, API Version: {api_version}")

        # --- File Processing ---
        processed_count = 0
        error_count = 0
        json_error_count = 0

        with open(synthetic_file, 'r', encoding='utf-8') as synth_f, \
             open(real_file, 'r', encoding='utf-8') as real_f, \
             open(output_file, 'w', encoding='utf-8') as out_f:

            logging.info(f"Starting processing files for LIVE inference: '{synthetic_file}' & '{real_file}'")
            logging.info(f"Output will be saved to: '{output_file}'")

            # Iterate through both files line by line simultaneously
            for line_num, (synth_line, real_line) in enumerate(zip(synth_f, real_f), 1):
                output_data = None # Ensure defined in scope
                try:
                    synthetic_record = json.loads(synth_line.strip())
                    real_record = json.loads(real_line.strip())
                    logging.debug(f"Successfully parsed JSON for line {line_num}.")

                    # Process the parsed records using the live function
                    output_data = process_single_record_live(
                        line_num=line_num,
                        synthetic_record=synthetic_record,
                        real_record=real_record,
                        system_prompt_content=system_prompt_content,
                        client=client
                    )

                    if output_data and output_data.get("error"):
                        error_count += 1
                    elif output_data and output_data.get("model_output") is not None:
                         processed_count += 1
                    else: # Case where processing didn't error but API failed/returned None
                         error_count +=1


                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON Decode Error in input files at line {line_num}: {json_err}")
                    json_error_count += 1
                    error_count += 1
                    output_data = { "line_number": line_num, "error": f"JSON Decode Error: {json_err}", "synthetic_line": synth_line.strip(), "real_line": real_line.strip() }
                except Exception as line_err:
                    logging.error(f"Unexpected error processing line {line_num} before API call: {line_err}", exc_info=True)
                    error_count += 1
                    output_data = { "line_number": line_num, "error": f"Unexpected line processing error: {line_err}" }

                # Write the result (or error) to the output file
                if output_data:
                    out_f.write(json.dumps(output_data) + '\n')
                else:
                     logging.error(f"Output data was unexpectedly None for line {line_num}")


            logging.info(f"LIVE processing complete.")
            logging.info(f"Successfully generated outputs for: {processed_count} records.")
            logging.info(f"Total errors encountered: {error_count} (including {json_error_count} JSON decode errors).")

    except FileNotFoundError:
        logging.critical("Exiting due to missing critical input file.")
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred in the LIVE pipeline: {e}", exc_info=True)


def prepare_batch_input(prompt_file, synthetic_file, real_file, batch_input_file, metadata_file):
    """
    Pipeline function for preparing BATCH inference input files.
    Generates two files: one for Azure Batch API, one for metadata.
    """
    logging.info("--- Running in BATCH Preparation Mode ---")
    try:
        # --- Initialization ---
        # No API key or client needed for batch prep
        system_prompt_content = read_prompt(prompt_file)
        logging.info("Read system prompt for batch preparation.")

        # --- File Processing ---
        prepared_count = 0
        error_count = 0
        json_error_count = 0

        with open(synthetic_file, 'r', encoding='utf-8') as synth_f, \
             open(real_file, 'r', encoding='utf-8') as real_f, \
             open(batch_input_file, 'w', encoding='utf-8') as batch_out_f, \
             open(metadata_file, 'w', encoding='utf-8') as meta_out_f:

            logging.info(f"Starting processing files for BATCH preparation: '{synthetic_file}' & '{real_file}'")
            logging.info(f"Azure Batch input will be saved to: '{batch_input_file}'")
            logging.info(f"Metadata will be saved to: '{metadata_file}'")

            # Iterate through both files line by line simultaneously
            for line_num, (synth_line, real_line) in enumerate(zip(synth_f, real_f), 1):
                metadata_output = None # Ensure defined
                batch_request = None
                try:
                    # Parse JSON data from each line
                    synthetic_record = json.loads(synth_line.strip())
                    real_record = json.loads(real_line.strip())
                    logging.debug(f"Successfully parsed JSON for line {line_num}.")

                    # --- Prepare Data for Both Output Files ---

                    # Extract real person text
                    abstract = real_record.get('Abstract', '') or ''
                    personal_life = real_record.get('Personal_Life', '') or ''
                    early_life = real_record.get('Early_Life', '') or ''
                    real_person_text = f"{abstract}\n{personal_life}\n{early_life}".strip()
                    if not real_person_text:
                        logging.warning(f"Real person text is empty for line {line_num}.")

                    # Format synthetic PII details string
                    synthetic_pii_details_str = json.dumps(synthetic_record, indent=2)

                    # Construct the User Prompt content
                    user_prompt_content = f"""**Inputs**:\nreal_wiki_inspiration_text:\n{real_person_text} \n synthetic_profile_json: \n {synthetic_pii_details_str}"""

                    # Use Unique ID from synthetic record as custom_id
                    custom_id = synthetic_record.get("Unique ID")
                    if not custom_id:
                        logging.warning(f"Missing 'Unique ID' in synthetic record at line {line_num}. Generating a UUID.")
                        custom_id = str(uuid.uuid4()) # Generate fallback ID

                    # --- Create Azure Batch Request Body ---
                    batch_request_body = {
                        "model": BATCH_MODEL_DEPLOYMENT, # Use BATCH model name
                        "messages": [
                            {"role": "system", "content": system_prompt_content},
                            {"role": "user", "content": user_prompt_content}
                        ],
                         # Add other parameters if supported/needed by batch API for your model
                         # "max_completion_tokens": MAX_TOKENS,
                         # "temperature": TEMPERATURE, # Optional: Check if batch supports these directly
                         # "top_p": TOP_P,
                    }

                    # --- Create Azure Batch Request JSON Line ---
                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/chat/completions", # Relative URL for the completions endpoint
                        "body": batch_request_body
                    }

                    # --- Create Metadata JSON Line ---
                    metadata_output = {
                        "line_number": line_num,
                        "custom_id": custom_id, # Link to the batch request
                        "synthetic_pii_input": synthetic_record,
                        "real_person_text_input": real_person_text,
                        "system_prompt": system_prompt_content, # Store prompts for reference
                        "user_prompt": user_prompt_content,
                        "model_parameters": {
                            "model": BATCH_MODEL_DEPLOYMENT, # Record intended model
                            # "temperature": TEMPERATURE,
                            # "top_p": TOP_P,
                            "max_tokens": MAX_TOKENS,
                        },
                        # No model_output or error fields here
                    }

                    # Write to output files
                    batch_out_f.write(json.dumps(batch_request) + '\n')
                    meta_out_f.write(json.dumps(metadata_output) + '\n')
                    prepared_count += 1
                    logging.debug(f"Successfully prepared batch/metadata for line {line_num} (custom_id: {custom_id}).")


                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON Decode Error in input files at line {line_num}: {json_err}")
                    json_error_count += 1
                    error_count += 1
                    # Log error only to metadata file for batch prep
                    metadata_output = {
                        "line_number": line_num,
                        "error": f"JSON Decode Error: {json_err}",
                        "synthetic_line": synth_line.strip(),
                        "real_line": real_line.strip()
                    }
                    meta_out_f.write(json.dumps(metadata_output) + '\n')
                except KeyError as ke:
                    logging.error(f"Missing expected key in input record at line {line_num}: {ke}")
                    error_count += 1
                    metadata_output = { "line_number": line_num, "error": f"Data processing error: Missing key '{ke}'" }
                    meta_out_f.write(json.dumps(metadata_output) + '\n')
                except Exception as line_err:
                    logging.error(f"Unexpected error processing line {line_num} during batch prep: {line_err}", exc_info=True)
                    error_count += 1
                    metadata_output = { "line_number": line_num, "error": f"Unexpected line processing error: {line_err}" }
                    meta_out_f.write(json.dumps(metadata_output) + '\n')

            logging.info(f"BATCH preparation complete.")
            logging.info(f"Successfully prepared records for: {prepared_count} inputs.")
            logging.info(f"Total errors encountered during preparation: {error_count} (including {json_error_count} JSON decode errors).")

    except FileNotFoundError:
        logging.critical("Exiting due to missing critical input file.")
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred in the BATCH preparation pipeline: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synthetic and real profiles using Azure OpenAI (Live or Batch Prep).")

    # Input Files
    parser.add_argument("--api-key-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Project/api_key.txt", help="Path to the file containing the Azure OpenAI API key.")
    parser.add_argument("--prompt-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/prompts/synthetic_article_gen_prompt.md", help="Path to the system prompt file.")
    parser.add_argument("--synthetic-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/synthetic_profiles_20250418_231425.jsonl", help="Path to the input synthetic profiles JSONL file.")
    parser.add_argument("--real-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/wikipedia_people_20250409_202759.jsonl", help="Path to the input real profiles JSONL file.")

    # Output Files / Mode
    parser.add_argument("--output-base", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/azure_batch_job", help="Base path and filename for output files (without extension). Timestamp and mode-specific suffixes will be added.")
    parser.add_argument("--mode", default='batch', choices=['live', 'batch'], required=True, help="Operation mode: 'live' for direct inference, 'batch' for preparing batch input files.")

    # Azure Configuration
    parser.add_argument("--azure-endpoint", default=DEFAULT_ENDPOINT, help="Azure OpenAI endpoint URL.")
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION, help="Azure OpenAI API version.")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the level.")

    args = parser.parse_args()

    # --- Setup Logging ---
    setup_logging(args.log_level)
    logging.info(f"Script arguments: {args}") # Log parsed arguments

    # --- Determine Output Filenames ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == 'live':
        output_file = f"{args.output_base}_{timestamp}.live.jsonl" # Changed extension to jsonl
    elif args.mode == 'batch':
        batch_input_file = f"{args.output_base}_{timestamp}.batch_input.jsonl"
        metadata_file = f"{args.output_base}_{timestamp}.metadata.jsonl"
    else:
        # This case should not be reached due to argparse choices
        logging.critical(f"Invalid mode selected: {args.mode}")
        exit(1)

    # --- Run Selected Pipeline ---
    if args.mode == 'live':
        run_live_pipeline(
            api_key_file=args.api_key_file,
            prompt_file=args.prompt_file,
            synthetic_file=args.synthetic_file,
            real_file=args.real_file,
            output_file=output_file, # Use mode-specific name
            azure_endpoint=args.azure_endpoint,
            api_version=args.api_version
        )
    elif args.mode == 'batch':
        prepare_batch_input(
            prompt_file=args.prompt_file,
            synthetic_file=args.synthetic_file,
            real_file=args.real_file,
            batch_input_file=batch_input_file, # Use mode-specific name
            metadata_file=metadata_file      # Use mode-specific name
        )

    logging.info("--- Script Finished ---")