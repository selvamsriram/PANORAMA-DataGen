import csv
import json
import random
import sys

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

def tsv_to_jsonl(input_tsv, num_rows, seed, output_jsonl, is_serialized=False):
    # Read the TSV file
    with open(input_tsv, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        rows = list(reader)
    
    # Deserialize rows if the TSV is serialized
    if is_serialized:
        for row in rows:
            for key, value in row.items():
                try:
                    row[key] = json.loads(value)  # Deserialize JSON strings
                except json.JSONDecodeError:
                    pass  # Keep the value as is if it's not JSON serialized
    
    # Shuffle the rows with the given seed
    random.seed(seed)
    random.shuffle(rows)
    
    # Select the first N rows
    selected_rows = rows[:num_rows]
    
    # Write to JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        for row in selected_rows:
            jsonl_file.write(json.dumps(row) + '\n')

if __name__ == "__main__":
    # Example usage
    input_tsv = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/synthetic_profiles_20250418_231425.tsv"
    num_rows = 60000
    seed = 800
    output_jsonl = "/Users/sriramselvam/Code/paraphrase_datagen/DataGeneration/Data/synthetic_profiles_20250418_231425.jsonl"
    is_serialized = True  # Set to True if the TSV is serialized
    
    tsv_to_jsonl(input_tsv, num_rows, seed, output_jsonl, is_serialized)
    print(f"JSONL file created at {output_jsonl}")