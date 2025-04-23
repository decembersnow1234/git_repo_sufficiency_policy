"""
Centralized pipeline for policy extraction and clustering from scientific abstracts
using enhanced NLP models and semantic clustering with PySpark and Spark NLP.
"""

import logging
from pyspark.sql import SparkSession
from modules.config import Config
from modules.data_processor import DataProcessor
from modules.policy_extractor import PolicyExtractor
from modules.policy_clusterer import PolicyClusterer
from modules.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("policy_pipeline.log")]
)
logger = logging.getLogger(__name__)

class PolicyPipeline:
    """Main pipeline orchestrating policy extraction and clustering using Spark NLP & PySpark"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configurations"""
        self.config = Config(config_path)
        self.spark = SparkSession.builder \
            .appName("PolicyPipeline") \
            .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:4.4.0") \
            .getOrCreate()

        self.processor = DataProcessor(self.config, self.spark)
        self.extractor = PolicyExtractor(self.config, self.spark)
        self.clusterer = PolicyClusterer(self.config, self.spark)
        self.visualizer = Visualizer(self.config, self.spark)

    def run(self, input_file: str):
        """Run the entire policy processing pipeline using Spark"""

        logger.info(f"Starting policy extraction pipeline for {input_file}")

        # Load and preprocess data
        df = self.processor.load_data(input_file)
        df = self.processor.preprocess_abstracts(df)

        # Extract policies
        df = self.extractor.extract_policies(df)

        # Perform clustering
        cluster_df = self.clusterer.cluster_policies(df)

        # Generate visualizations
        self.visualizer.visualize_clusters(df, cluster_df)

        # Save final output
        self.processor.save_data(cluster_df, "processed_policies.parquet")

        logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and cluster policy statements using PySpark & Spark NLP")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Input file with abstracts (CSV, JSON, or Parquet)")

    args = parser.parse_args()

    pipeline = PolicyPipeline(args.config)
    pipeline.run(args.input)
