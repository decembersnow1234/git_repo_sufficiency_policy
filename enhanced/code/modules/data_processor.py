import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import spacy
import yaml
from tqdm import tqdm

from modules.config import Config

class DataProcessor:
    """Data loading and preprocessing for scientific abstracts"""

    def __init__(self, config: Config):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")

    def load_data(self, input_file: str) -> pd.DataFrame:
        """Load data from CSV, JSON, or Excel file"""
        file_path = Path(input_file) if not (self.config.data_dir / input_file).exists() else self.config.data_dir / input_file

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        file_extension = file_path.suffix.lower()
        loaders = {
            '.csv': pd.read_csv,
            '.json': pd.read_json,
            '.xls': pd.read_excel,
            '.xlsx': pd.read_excel
        }

        if file_extension not in loaders:
            raise ValueError(f"Unsupported file format: {file_extension}")

        df = loaders[file_extension](file_path)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df

    def preprocess_abstracts(self, abstracts: List[str]) -> List[Dict]:
        """Preprocess abstracts with spaCy for enhanced NLP analysis"""
        processed_docs = []
        
        for doc in tqdm([self.nlp(abstract) for abstract in abstracts if isinstance(abstract, str) and abstract.strip()], desc="Preprocessing abstracts"):
            processed_docs.append({
                'text': doc.text,
                'sentences': [sent.text for sent in doc.sents],
                'entities': [{'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char} for ent in doc.ents],
                'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
                'tokens': [{'text': token.text, 'pos': token.pos_, 'dep': token.dep_} for token in doc if not token.is_stop and not token.is_punct]
            })
        
        logging.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs
