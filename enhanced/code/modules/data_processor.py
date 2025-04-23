import logging
from venv import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, FloatType
import spacy
import pandas as pd
from pathlib import Path
from typing import Dict, List
from pyspark.sql.types import StructType, StructField

from modules.config import Config

class DataProcessor:
    """Data loading and preprocessing for scientific abstracts using PySpark"""

    def __init__(self, config: Config, spark: SparkSession):
        """Initialize DataProcessor with Spark & SpaCy"""
        self.config = config
        self.spark = spark
        self.nlp = spacy.load("en_core_web_sm")  # Load SpaCy globally OUTSIDE Spark transformations

    def load_data(self, input_file: str) -> pd.DataFrame:
    """Load data from CSV, JSON, or Excel file in chunks if necessary."""
    file_path = self.config.data_dir / input_file
    resolved_path = file_path.resolve()
    print(f"Attempting to load file from: {resolved_path}")
    
    if not file_path.exists():
        file_path = Path(input_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, chunksize=10000)
        df = pd.concat(df)
    elif file_path.suffix.lower() == '.json':
        df = pd.read_json(file_path)
    elif file_path.suffix.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logging.info(f"Loaded {len(df)} records from {file_path}")
    return df

    from concurrent.futures import ProcessPoolExecutor

def preprocess_abstracts(self, abstracts: List[str]) -> List[Dict]:
    """Preprocess abstracts with spaCy for enhanced NLP analysis using parallel processing."""
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        processed_docs = list(tqdm(executor.map(self._process_single_abstract, abstracts), total=len(abstracts), desc="Preprocessing abstracts"))
    
    logger.info(f"Preprocessed {len(processed_docs)} documents")
    return processed_docs

def _process_single_abstract(self, abstract: str) -> Dict:
    """Helper function to process a single abstract."""
    if not isinstance(abstract, str) or not abstract.strip():
        return {}
    
    doc = self.nlp(abstract)
    processed_doc = {
        'text': abstract,
        'sentences': [sent.text for sent in doc.sents],
        'entities': [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            for ent in doc.ents
        ],
        'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
        'tokens': [
            {
                'text': token.text,
                'pos': token.pos_,
                'dep': token.dep_
            }
            for token in doc if not token.is_stop and not token.is_punct
        ]
    }
    return processed_doc

    def save_data(self, df, output_file: str):
        """Save processed data efficiently using Parquet"""
        output_path = str(self.config.output_dir / output_file)
        df.write.parquet(output_path, mode="overwrite")
        logging.info(f"Saved processed data to {output_path}")
