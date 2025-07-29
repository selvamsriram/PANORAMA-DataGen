#!/usr/bin/env python3
"""
Entity Contamination Filter for PANORAMA Data Generation

This script detects when entities from real_person_text_input appear in 
generated_synthetic_passage, indicating potential contamination of synthetic data
with real person information.

Usage:
    python entity_contamination_filter.py --input_file data.jsonl --output_file contamination_report.tsv
"""

import json
import argparse
import logging
import re
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
import spacy
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityContaminationFilter:
    """Detects entity contamination between real person text and synthetic passages."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the filter with spaCy model.
        
        Args:
            spacy_model: spaCy model to use for entity extraction
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found. Please install it with:")
            logger.error(f"python -m spacy download {spacy_model}")
            raise
        
        # Define entity types to check for contamination
        self.target_entities = {
            'PERSON', 'ORG', 'LOC', 'FAC', 'PRODUCT'
        }
        
        # Custom patterns for additional entity extraction
        self.custom_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'ZIP_CODE': r'\b\d{5}(?:-\d{4})?\b',
            'CREDIT_CARD': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        }
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract entities from text using spaCy and custom patterns.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary mapping entity types to sets of entity values
        """
        if not text or not text.strip():
            return {}
        
        entities = {}
        
        # Extract entities using spaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in self.target_entities:
                entity_type = ent.label_
                entity_value = ent.text.strip()
                if entity_type not in entities:
                    entities[entity_type] = set()
                entities[entity_type].add(entity_value)
        
        # Extract entities using custom patterns
        for pattern_name, pattern in self.custom_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if pattern_name not in entities:
                    entities[pattern_name] = set()
                # Handle different match formats
                for match in matches:
                    if isinstance(match, tuple):
                        entity_value = ''.join(match)
                    else:
                        entity_value = match
                    entities[pattern_name].add(entity_value)
        
        # Extract names (common pattern: "First Last")
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        name_matches = re.findall(name_pattern, text)
        if name_matches:
            if 'PERSON' not in entities:
                entities['PERSON'] = set()
            entities['PERSON'].update(name_matches)
        
        return entities
    
    def find_contaminations(self, real_entities: Dict[str, Set[str]], 
                          synthetic_entities: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
        """
        Find overlapping entities between real and synthetic text.
        
        Args:
            real_entities: Entities from real_person_text_input
            synthetic_entities: Entities from generated_synthetic_passage
            
        Returns:
            List of tuples (entity_type, entity_value) for overlapping entities
        """
        contaminations = []
        
        for entity_type in real_entities:
            if entity_type in synthetic_entities:
                real_values = real_entities[entity_type]
                synthetic_values = synthetic_entities[entity_type]
                
                # Check for exact matches
                overlaps = real_values.intersection(synthetic_values)
                for entity_value in overlaps:
                    contaminations.append((entity_type, entity_value))
                
                # Check for partial matches (substring containment)
                for real_value in real_values:
                    for synthetic_value in synthetic_values:
                        if (real_value.lower() in synthetic_value.lower() or 
                            synthetic_value.lower() in real_value.lower()):
                            if (entity_type, real_value) not in contaminations:
                                contaminations.append((entity_type, real_value))
        
        return contaminations
    
    def process_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single record to detect entity contamination.
        
        Args:
            record: JSON record from the input file
            
        Returns:
            List of contamination records (empty if no contamination found)
        """
        contamination_records = []
        
        # Extract the required fields
        unique_id = record.get('synthetic_pii_input', {}).get('Unique ID', 'Unknown')
        real_text = record.get('real_person_text_input', '')
        synthetic_text = record.get('SyntheticPassageGeneration', {}).get('generated_synthetic_passage', '')
        
        if not real_text or not synthetic_text:
            return contamination_records
        
        # Extract entities from both texts
        real_entities = self.extract_entities(real_text)
        synthetic_entities = self.extract_entities(synthetic_text)
        
        # Find contaminations
        contaminations = self.find_contaminations(real_entities, synthetic_entities)
        
        # Create contamination records
        for entity_type, entity_value in contaminations:
            # Convert sets to lists for JSON serialization
            real_entities_serializable = {k: list(v) for k, v in real_entities.items()}
            synthetic_entities_serializable = {k: list(v) for k, v in synthetic_entities.items()}
            
            contamination_record = {
                'unique_id': unique_id,
                'entity_type': entity_type,
                'entity_value': entity_value,
                'complete_record': json.dumps(record, ensure_ascii=False),
                'real_entities': json.dumps(real_entities_serializable, ensure_ascii=False),
                'synthetic_entities': json.dumps(synthetic_entities_serializable, ensure_ascii=False)
            }
            contamination_records.append(contamination_record)
        
        return contamination_records
    
    def process_file(self, input_file: str, output_file: str) -> None:
        """
        Process the entire JSONL file and generate contamination report.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output TSV file
        """
        logger.info(f"Processing file: {input_file}")
        
        all_contaminations = []
        total_records = 0
        contaminated_records = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    total_records += 1
                    
                    contaminations = self.process_record(record)
                    if contaminations:
                        all_contaminations.extend(contaminations)
                        contaminated_records += 1
                    
                    if total_records % 1000 == 0:
                        logger.info(f"Processed {total_records} records...")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
        
        # Write results to TSV file
        if all_contaminations:
            df = pd.DataFrame(all_contaminations)
            df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
            logger.info(f"Found {len(all_contaminations)} contaminations in {contaminated_records} records")
            logger.info(f"Results written to: {output_file}")
        else:
            logger.info("No contaminations found")
        
        logger.info(f"Total records processed: {total_records}")
        logger.info(f"Records with contamination: {contaminated_records}")


def main():
    """Main function to run the entity contamination filter."""
    parser = argparse.ArgumentParser(
        description="Detect entity contamination in synthetic data generation"
    )
    parser.add_argument(
        '--input_file', 
        required=True,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output_file', 
        required=True,
        help='Path to output TSV file'
    )
    parser.add_argument(
        '--spacy_model',
        default='en_core_web_sm',
        help='spaCy model to use for entity extraction (default: en_core_web_sm)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    try:
        # Initialize and run the filter
        filter_obj = EntityContaminationFilter(args.spacy_model)
        filter_obj.process_file(args.input_file, args.output_file)
        return 0
        
    except Exception as e:
        logger.error(f"Error running entity contamination filter: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 