import logging
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

    def load_data(self, input_file: str):
        """Load data efficiently using PySpark and Pandas for Excel files"""
        file_path = str(self.config.data_dir / input_file)

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            # Read Excel with Pandas
            df_pandas = pd.read_excel(file_path)

            # Convert integer columns to float before passing to PySpark
            for col in df_pandas.select_dtypes(include=["int64"]).columns:
                df_pandas[col] = df_pandas[col].astype(float)  # Ensure all int64 values are float

            # Define schema for numeric columns only (leave others dynamic)
            schema = StructType([
                StructField(col, FloatType(), True) if df_pandas[col].dtype == "float64" else StructField(col, StringType(), True)
                for col in df_pandas.columns
            ])

            # Convert Pandas DataFrame to Spark DataFrame
            print(df_pandas.columns)
            df = self.spark.createDataFrame(df_pandas)

        elif file_path.endswith(".csv"):
            df = self.spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)
        elif file_path.endswith(".json"):
            df = self.spark.read.json(file_path)
        elif file_path.endswith(".parquet"):
            df = self.spark.read.parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        logging.info(f"Loaded {df.count()} records from {file_path}")
        return df

    def preprocess_abstracts(self, df):
        """Process abstracts using SpaCy (parallelized with Spark UDF)"""

        # Ensure "abstract" column exists
        if "abstract" not in df.columns:
            raise ValueError("DataFrame does not contain column 'abstract'")

        def extract_entities(text: str):
            """Extract entities using SpaCy **inside** the UDF (to prevent serialization issues)"""
            nlp_local = spacy.load("en_core_web_sm")  # Load SpaCy inside function to avoid Spark serialization errors
            doc = nlp_local(text)
            return [ent.text for ent in doc.ents]

        # Convert extract_entities function to a Spark UDF
        entity_udf = udf(extract_entities, ArrayType(StringType()))

        # Apply UDF using the correct column reference
        df = df.withColumn("entities", entity_udf(col("abstract")))  # Use col() for Spark column reference

        logging.info("Finished processing abstracts using SpaCy and Spark")
        return df

    def save_data(self, df, output_file: str):
        """Save processed data efficiently using Parquet"""
        output_path = str(self.config.output_dir / output_file)
        df.write.parquet(output_path, mode="overwrite")
        logging.info(f"Saved processed data to {output_path}")
