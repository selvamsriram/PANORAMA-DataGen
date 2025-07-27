from datetime import datetime
import os
import json
import argparse
import logging
import tiktoken
from openai import AzureOpenAI, OpenAIError


# --- Configuration ---
# Azure OpenAI endpoint details (can be overridden by arguments)
# Use the endpoint relevant to your deployment
DEFAULT_ENDPOINT = "" # e.g., "https://<your-resource-name>.openai.azure.com/"
DEFAULT_API_VERSION = "2024-12-01-preview" # Use a specific, tested version (check Azure documentation)
# Model deployment name for live inference
LIVE_MODEL_DEPLOYMENT = "o3-mini" # <<< PLEASE UPDATE THIS (e.g., gpt-4)

# Recommended generation parameters
MAX_TOKENS = 3500 # Adjust as needed

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
    Calls the Azure OpenAI Chat Completions API (Live Mode), logs token usage on the response,
    and handles errors. Returns the content string or None on failure.
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
            # temperature, top_p, etc. can be added here
        )

        # Validate structure
        if not (response and response.choices and len(response.choices) > 0):
            logging.warning("Received an empty or invalid response structure from Azure OpenAI.")
            return None

        choice = response.choices[0]
        if not (choice.message and choice.message.content):
            logging.warning("Valid response structure, but message content is missing.")
            return None

        model_output = choice.message.content

        # --- New: count tokens in the response ---
        try:
            # get the encoder for this model (falls back if model name not recognized)
            encoding = tiktoken.encoding_for_model(model_deployment)
        except KeyError:
            # fallback to a default encoding
            encoding = tiktoken.get_encoding("gpt2")

        tokens = encoding.encode(model_output)
        num_tokens = len(tokens)
        logging.info(f"Response token count: {num_tokens}")
        logging.debug(f"Response tokens: {tokens[:10]}... (total {num_tokens})")
        # ------------------------------------------

        logging.info("Successfully received response from Azure OpenAI.")
        logging.debug(f"Response length (chars): {len(model_output)}")
        return model_output

    except OpenAIError as api_err:
        if hasattr(api_err, 'status_code'):
            logging.error(f"Azure OpenAI API Error (Status {api_err.status_code}): {api_err}")
        else:
            logging.error(f"Azure OpenAI API Error: {api_err}")
        if hasattr(api_err, 'response') and api_err.response:
            logging.error(f"API Response Body: {api_err.response.text}")
        return None

    except Exception as call_err:
        logging.error(f"Unexpected error during API call: {call_err}", exc_info=True)
        return None

# --- Pipeline Function ---

def process_single_record_live(line_num, record, system_prompt_content, client, model_deployment, max_tokens_param):
    """
    Processes a single record from the input JSONL for LIVE inferencing.
    Returns a dictionary containing inputs, parameters, and output/error.
    """
    logging.debug(f"Processing record (Live): {line_num}")
    # Initialize output structure
    output_data = {
        "system_prompt": system_prompt_content,
        "user_prompt": None,
        "model_parameters": {
            "model": model_deployment, # Use passed model name
            "max_tokens": max_tokens_param,
        },
        "model_output": None,
        "error": None
    }

    try:
        # --- Extract data and construct user prompt ---
        # Use .get() for safer access, providing default empty values or dicts
        synthetic_pii = record.get('synthetic_pii_input', {})
        passage_gen_info = record.get('SyntheticPassageGeneration', {})
        generated_passage = passage_gen_info.get('generated_synthetic_passage', '')

        # Convert synthetic_pii (which is expected to be a dict) to a JSON string for the prompt
        synthetic_pii_str = json.dumps(synthetic_pii, indent=2) if synthetic_pii else "{}"

        if not synthetic_pii:
            logging.warning(f"Field 'synthetic_pii_input' is missing or empty in record at line {line_num}.")
        if not generated_passage:
            logging.warning(f"Field 'SyntheticPassageGeneration.generated_synthetic_passage' is missing or empty in record at line {line_num}.")

        # Construct the User Prompt by joining the two fields
        # Adjust the formatting/separator as needed for your prompt's requirements
        user_prompt_content = f"**Synthetic Profile Json**:\n{synthetic_pii_str}\n\n**Synthetic Persona Article**:\n{generated_passage}"
        output_data["user_prompt"] = user_prompt_content
        logging.debug(f"Constructed user prompt (length: {len(user_prompt_content)}) for line {line_num}.")

        # --- Call Azure OpenAI ---
        model_output = call_azure_openai(
            client=client,
            system_prompt=system_prompt_content,
            user_prompt=user_prompt_content,
            model_deployment=model_deployment, # Use passed model
            max_tokens=max_tokens_param
        )

        if model_output is not None:
            output_data["model_output"] = model_output
            logging.info(f"Successfully processed record {line_num} (Live).")
        else:
            output_data["error"] = "API call failed or returned empty/invalid response. See logs."
            logging.warning(f"API call failed for record {line_num} (Live).")

    except json.JSONDecodeError as json_err:
        # This error would typically happen when reading the input file, handled in the main loop
        logging.error(f"Error decoding JSON within the record at line {line_num} (should not happen here): {json_err}")
        output_data["error"] = f"Internal JSON processing error: {json_err}"
    except KeyError as ke:
         # Should be less likely with .get(), but good practice
         logging.error(f"Missing expected key structure in input record at line {line_num}: {ke}")
         output_data["error"] = f"Data processing error: Missing key structure '{ke}'"
    except Exception as proc_err:
        logging.error(f"Error processing data for record {line_num} (Live): {proc_err}", exc_info=True)
        output_data["error"] = f"Data processing error: {proc_err}"

    record ["SyntheticTrainingData"] = output_data
    return record


def run_live_pipeline(api_key_file, prompt_file, input_file, output_file, azure_endpoint, api_version, model_deployment, max_tokens_param):
    """
    Pipeline function for LIVE inferencing using a single JSONL input file.
    """
    logging.info("--- Running Azure OpenAI Live Inference ---")
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
        logging.info(f"Initialized AzureOpenAI client. Endpoint: {azure_endpoint}, API Version: {api_version}, Deployment: {model_deployment}")

        # --- File Processing ---
        processed_count = 0
        error_count = 0
        json_error_count = 0

        with open(input_file, 'r', encoding='utf-8') as in_f, \
             open(output_file, 'w', encoding='utf-8') as out_f:

            logging.info(f"Starting processing file: '{input_file}'")
            logging.info(f"Output will be saved to: '{output_file}'")

            # Iterate through the input file line by line
            for line_num, line in enumerate(in_f, 1):
                output_data = None  # Ensure defined in scope
                try:
                    record = json.loads(line.strip())
                    logging.debug(f"Successfully parsed JSON for line {line_num}.")

                    # Process the parsed record using the live function
                    record = process_single_record_live(
                        line_num=line_num,
                        record=record,
                        system_prompt_content=system_prompt_content,
                        client=client,
                        model_deployment=model_deployment,  # Pass deployment name
                        max_tokens_param=max_tokens_param  # Pass max tokens
                    )
                    # Extract SyntheticTrainingData from the record
                    synthetic_training_data = record.get("SyntheticTrainingData", {})

                    if synthetic_training_data.get("error"):
                        error_count += 1
                    elif synthetic_training_data.get("model_output") is not None:
                        processed_count += 1
                    else:
                        error_count += 1
                        if not synthetic_training_data.get("error"):
                            synthetic_training_data["error"] = "Processing succeeded but API call failed or returned no content."

                    output_data = record

                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON Decode Error in input file at line {line_num}: {json_err}")
                    json_error_count += 1
                    error_count += 1
                    # Create error entry for output
                    output_data = {
                        "line_number": line_num,
                        "error": f"JSON Decode Error: {json_err}",
                        "original_line": line.strip()  # Include the problematic line
                    }
                except Exception as line_err:
                    logging.error(f"Unexpected error processing line {line_num} before API call: {line_err}", exc_info=True)
                    error_count += 1
                    output_data = {
                        "line_number": line_num,
                        "error": f"Unexpected line processing error: {line_err}"
                    }

                # Write the updated record (or error) to the output file
                if output_data:
                    out_f.write(json.dumps(output_data) + '\n')
                else:
                    logging.error(f"Output data was unexpectedly None for line {line_num}. Writing basic error.")
                    error_entry = {"line_number": line_num, "error": "Unknown processing error: output_data was None."}
                    out_f.write(json.dumps(error_entry) + '\n')
                    error_count += 1


            logging.info(f"Live processing complete.")
            logging.info(f"Successfully generated outputs for: {processed_count} records.")
            logging.info(f"Total errors encountered: {error_count} (including {json_error_count} JSON decode errors).")

    except FileNotFoundError as fnf_err:
        logging.critical(f"Exiting due to missing critical input file: {fnf_err}")
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred in the live pipeline: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    # Simplified parser for live mode only
    parser = argparse.ArgumentParser(description="Process JSONL records using Azure OpenAI Live Inference.")

    # Input Files
    parser.add_argument("--api-key-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Project/api_key.txt", help="Path to the file containing the Azure OpenAI API key.")
    parser.add_argument("--prompt-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/prompts/synthetic_online_text_gen_prompt.md", help="Path to the system prompt file.")
    parser.add_argument("--input-file", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/extracted_passages_azure_results.10K.03.jsonl", help="Path to the extracted and generated passages JSONL file.")

    # Output File
    parser.add_argument("--output-base", default="/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/Azure_Synthetic_Data_10K.03.", help="Base path and filename for the output file (without extension). Timestamp and '.live_output.jsonl' will be added.")

    # Azure Configuration - REMEMBER TO UPDATE DEFAULTS OR PROVIDE VIA ARGUMENTS
    parser.add_argument("--azure-endpoint", default=DEFAULT_ENDPOINT, help="Azure OpenAI endpoint URL.")
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION, help="Azure OpenAI API version.")
    parser.add_argument("--deployment", default=LIVE_MODEL_DEPLOYMENT, help="Name of the Azure OpenAI deployment for live inference.") # Renamed from --live-deployment
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Maximum number of tokens to generate in the completion.") # Added max_tokens argument

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level.")

    args = parser.parse_args()

    # --- Setup Logging ---
    setup_logging(args.log_level)
    logging.info(f"Script arguments: {args}") # Log parsed arguments

    # --- Create Output Directory if it doesn't exist ---
    output_dir = os.path.dirname(args.output_base)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Failed to create output directory '{output_dir}': {e}")
            exit(1) # Exit if directory creation fails

    # --- Determine Output Filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_base}_{timestamp}.live_output.jsonl"

    # --- Run Live Pipeline ---
    run_live_pipeline(
        api_key_file=args.api_key_file,
        prompt_file=args.prompt_file,
        input_file=args.input_file,
        output_file=output_file,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        model_deployment=args.deployment, # Pass deployment name from args
        max_tokens_param=args.max_tokens # Pass max tokens from args
    )

    logging.info("--- Script Finished ---")
