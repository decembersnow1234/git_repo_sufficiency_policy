#!/usr/bin/env python
"""
Script to run the policy extraction and clustering pipeline using Spark NLP & PySpark.
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pipeline.policy_pipeline import PolicyPipeline
from sparknlp.pretrained import PretrainedPipeline

# Correcting module path handling for cross-platform compatibility
MODULES_PATH = "/content/git_repo_sufficiency_policy/enhanced/code"
if not os.path.exists(MODULES_PATH):
    print(f"❌ ERROR: Modules directory '{MODULES_PATH}' not found!")
    sys.exit(1)

sys.path.append(MODULES_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"pipeline_run_{int(time.time())}.log")
    ]
)
logger = logging.getLogger("pipeline_runner")

def verify_available_models():
    """Check and list available pre-trained models in Spark NLP"""
    print("\n🔍 Available Pretrained Pipelines:\n")
    print(PretrainedPipeline.available_pipelines())

def main():
    """Run the policy processing pipeline"""
    parser = argparse.ArgumentParser(description="Extract and cluster policy statements using PySpark & Spark NLP")
    parser.add_argument("--config", default="/content/git_repo_sufficiency_policy/enhanced/code/config.yaml", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Input file with abstracts (CSV, JSON, or Parquet)")
    args = parser.parse_args()

    # Ensure the config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    # ✅ Verify available models before running
    verify_available_models()

    # ✅ Initialize Spark session
    try:
        spark = SparkSession.builder \
            .appName("PolicyPipeline") \
            .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.0") \
            .getOrCreate()
        print("✅ Spark initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Spark initialization failed: {e}")
        sys.exit(1)  # Exit if Spark fails

    # Run the policy pipeline
    try:
        pipeline = PolicyPipeline(config_path, spark)
        pipeline.run(args.input)
        logger.info("✅ Pipeline execution completed successfully!")
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
