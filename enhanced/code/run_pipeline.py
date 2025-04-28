#!/usr/bin/env python
# run_pipeline.py
"""
Script to run the policy extraction and clustering pipeline using Spark NLP & PySpark.
"""

import argparse
import logging
import json
import yaml
import sys
import os
import time
from pathlib import Path

# Add module paths dynamically
sys.path.append("git_repo_sufficiency_policy/enhanced/code")

from pyspark.sql import SparkSession
from modules.config import Config
from modules.data_processor import DataProcessor
from modules.policy_extractor import PolicyExtractor
from modules.policy_clusterer import PolicyClusterer
from modules.visualizer import Visualizer
from pipeline.policy_pipeline import PolicyPipeline

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"pipeline_run_{int(time.time())}.log")
        ]
    )
    return logging.getLogger("pipeline_runner")

def create_default_config(logger):
    """Create a default configuration file if none exists"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        default_config = {
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
                "model_dir": "models"
            },
            "models": {
                "policy_extractor": "classifierdl_bertwikiner_policy",
                "impact_classifier": "classifierdl_impact",
                "semantic_model": "all-mpnet-base-v2"
            },
            "parameters": {
                "min_sentence_length": 5,
                "policy_confidence_threshold": 0.3,
                "max_clusters": 20,
                "min_cluster_size": 5,
                "clustering_method": "hierarchical"
            }
        }
        with open(config_path, 'w') as config_file:
            yaml.dump(default_config, config_file, default_flow_style=False)
        logger.info(f"Created default configuration file at {config_path}")

def main():
    logger = setup_logging()
    create_default_config(logger)
    
    parser = argparse.ArgumentParser(description="Run the policy extraction and clustering pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Input file with abstracts (CSV or Parquet)")
    parser.add_argument("--skip-to", choices=['load', 'extract', 'classify', 'cluster', 'visualize'],
                        help="Skip to a specific pipeline stage using checkpoints")
    args = parser.parse_args()

    # Initialize Spark Session with Spark NLP
    spark = SparkSession.builder \
        .appName("PolicyPipeline") \
        .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:4.4.0") \
        .getOrCreate()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    pipeline = PolicyPipeline(config_path, spark)
    pipeline.run(args.input)

if __name__ == "__main__":
    main()
