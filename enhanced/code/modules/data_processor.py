import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType, StringType, StructType, StructField, FloatType
)
import spacy
import pandas as pd
from pathlib import Path
from typing import Dict, List
from modules.config import Config

class DataProcessor:
    """Data loading and preprocessing for scientific abstracts using PySpark"""

    def __init__(self, config: Config, spark: SparkSession):
        """Initialize DataProcessor with Spark & SpaCy"""
        self.config = config
        self.spark = spark
        self.nlp = spacy.load("en_core_web_sm")  # Load SpaCy globally

    def load_data(self, input_file: str) -> pd.DataFrame:
        """Load data from CSV, JSON, or Excel files efficiently."""
        file_path = Path(self.config.data_dir) / input_file
        resolved_path = file_path.resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {resolved_path}")

        logging.info(f"Loading file: {resolved_path}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logging.info(f"Loaded {len(df)} records from {resolved_path}")
        return df

    def preprocess_abstracts(self, abstracts: List[str]) -> List[Dict]:
        """Preprocess abstracts using spaCy for NLP analysis."""
        processed_docs = [self._process_single_abstract(abstract) for abstract in abstracts]
        logging.info(f"Preprocessed {len(processed_docs)} abstracts")
        return processed_docs

    def _process_single_abstract(self, abstract: str) -> Dict:
        """Helper function to process a single abstract."""
        if not isinstance(abstract, str) or not abstract.strip():
            return {}

        doc = self.nlp(abstract)
        return {
            'text': abstract,
            'sentences': [sent.text for sent in doc.sents],
            'entities': [
                {'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char}
                for ent in doc.ents
            ],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'tokens': [
                {'text': token.text, 'pos': token.pos_, 'dep': token.dep_}
                for token in doc if not token.is_stop and not token.is_punct
            ]
        }

    def save_data(self, df, output_file: str):
        """Save processed data using Parquet format."""
        output_path = str(Path(self.config.output_dir) / output_file)
        df.write.parquet(output_path, mode="overwrite")
        logging.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = Config("config.yaml")
    spark = SparkSession.builder.appName("DataProcessor").getOrCreate()
    
    processor = DataProcessor(config, spark)
    
    input_file = "input.csv"
    df = processor.load_data(input_file)
    abstracts = df['abstract'].tolist() if 'abstract' in df.columns else df['text'].tolist()
    processed_docs = processor.preprocess_abstracts(abstracts)
    
    schema = StructType([
        StructField("text", StringType(), True),
        StructField("sentences", ArrayType(StringType()), True),
        StructField("entities", ArrayType(
            StructType([
                StructField("text", StringType(), True),
                StructField("label", StringType(), True),
                StructField("start", FloatType(), True),
                StructField("end", FloatType(), True)
            ])
        ), True),
        StructField("noun_chunks", ArrayType(StringType()), True),
        StructField("tokens", ArrayType(
            StructType([
                StructField("text", StringType(), True),
                StructField("pos", StringType(), True),
                StructField("dep", StringType(), True)
            ])
        ), True)
    ])
    
    processed_df = spark.createDataFrame(processed_docs, schema)
    processor.save_data(processed_df, "processed_data.parquet")
