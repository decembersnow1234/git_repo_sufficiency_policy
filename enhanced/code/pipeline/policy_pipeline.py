"""
Centralized pipeline for policy extraction and clustering from scientific abstracts
with enhanced NLP models and semantic clustering using PySpark.
"""

import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import spacy
import yaml
from pathlib import Path
from modules.config import Config
from modules.data_processor import DataProcessor
from modules.policy_extractor import PolicyExtractor
from modules.policy_clusterer import PolicyClusterer
from modules.visualizer import Visualizer
from pyspark.ml.clustering import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("policy_pipeline.log")]
)
logger = logging.getLogger(__name__)

class PolicyPipeline:
    """Main pipeline orchestrating the policy extraction and clustering process"""

    def __init__(self, config_path: str = "config.yaml", logger=None):  # Add logger argument
        """Initialize the pipeline with configuration"""
        self.config = Config(config_path)
        self.logger = logger if logger else logging.getLogger(__name__)  # Assign logger
        self.spark = SparkSession.builder.appName("PolicyPipeline").getOrCreate()
        self.processor = DataProcessor(self.config, self.spark)
        self.extractor = PolicyExtractor(self.config, self.spark)
        self.clusterer = PolicyClusterer(self.config, self.spark)
        self.visualizer = Visualizer(self.config)

    def run(self, input_file: str):
        """Run the full pipeline with PySpark"""
        logger.info(f"Starting policy extraction pipeline on {input_file}")

        # Load data
        df = self.processor.load_data(input_file)

        # Preprocess abstracts (Spark UDF)
        df = self.processor.preprocess_abstracts(df)

        # Extract policies (Spark UDF)
        df = self.extractor.extract_policies(df)

        # Perform clustering
        df = self.clusterer.cluster_policies(df)

        # Visualize results
        self.visualizer.visualize_clusters(df)

        # Save final output
        self.processor.save_data(df, "processed_policies.parquet")
        logger.info(f"Pipeline completed successfully.")

class PolicyClusterer:
    """Cluster policy statements efficiently using Spark MLlib"""

    def __init__(self, config, spark):
        self.config = config
        self.spark = spark

    def cluster_policies(self, df):
        """Use KMeans clustering from Spark MLlib"""
        kmeans = KMeans(featuresCol="features", k=10)
        model = kmeans.fit(df)
        df = model.transform(df)
        return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and cluster policy statements using PySpark")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Input file with abstracts (CSV, JSON, or Parquet)")

    args = parser.parse_args()

    pipeline = PolicyPipeline(args.config)
    pipeline.run(args.input)
