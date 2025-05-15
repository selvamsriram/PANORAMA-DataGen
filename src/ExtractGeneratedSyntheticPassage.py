import json
import re

input_file = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/generated_passages_azure_results.10K.jsonl"   # change to your filename/path
output_file = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/extracted_passages_azure_results.10K.jsonl" # output filename

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        
        # Gather fields into new object
        synthetic_passage = {
            "system_prompt": record.get("system_prompt"),
            "user_prompt": record.get("user_prompt"),
            "model_parameters": record.get("model_parameters"),
            "model_output": record.get("model_output")
        }
        
        # If model_output exists extract generated_synthetic_passage
        model_output = synthetic_passage.get("model_output")
        if model_output:
            match = re.search(r"\[Synthetic Article\](.*?)\[Real Person Text Usage Notes\]", model_output, re.DOTALL)
            if match:
                generated_text = match.group(1).strip()
            else:
                generated_text = ""
        else:
            generated_text = ""
        
        if generated_text == "":
            print(f"Warning: No generated text found for record: {record.get('id')}")
            continue
        
        synthetic_passage["generated_synthetic_passage"] = generated_text
        
        # Insert the new object into the record
        record["SyntheticPassageGeneration"] = synthetic_passage
        
        # Remove the original keys from the record
        for key in ["system_prompt", "user_prompt", "model_parameters", "model_output"]:
            record.pop(key, None)
        
        # Write updated record to output file as JSONL
        outfile.write(json.dumps(record) + "\n")