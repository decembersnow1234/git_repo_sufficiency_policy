#!/usr/bin/env python
# run_pipeline.py
"""
Script to run the policy extraction and clustering pipeline.
ain"""
import argparse
import logging
from pathlib import Path
import yaml
import sys
import timefrom policy_pipeline import PolicyPipeline, DataProcessor, PolicyExtractor, PolicyClusterer, Visualizer, Config

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

def create_default_config():
    """Create a default configuration file if none exists"""
    default_config = {
        "paths": {
            "data_dir": "data",
            "output_dir": "output",
            "model_dir": "models"
        },
        "models": {
            "policy_extractor": "roberta-base",
            "impact_classifier": "distilbert-base-uncased",
            "semantic_model": "all-mpnet-base-v2"
        },
        "parameters": {
            "min_sentence_length": 5,
            "policy_confidence_threshold": 0.3,
            "max_clusters": 20,
            "min_cluster_size": 5,
            "clustering_method": "hierarchical"
        },
        "extraction": {
            "policy_keywords": [
                "policy", "regulation", "legislation", "law", "rule",
                "guideline", "framework", "program", "initiative",
                "strategy", "plan", "measure", "intervention",
                "approach", "mechanism", "instrument", "proposal",
                "recommendation", "action", "should", "must", "need", "require"
            ],
            "policy_patterns": [
                "MODAL VERB ACTION_VERB",
                "RECOMMEND THAT",
                "SUGGEST THAT",
                "POLICY TO VERB"
            ]
        },
        "impact": {
            "positive_keywords": [
                "improve", "increase", "enhance", "benefit",
                "positive", "advance", "strengthen", "promote"
            ],
            "negative_keywords": [
                "reduce", "decrease", "limit", "restrict",
                "negative", "harmful", "adverse", "worsen"
            ]
        },
        "visualization": {
            "colors": {
                "positive": "#2ecc71",
                "neutral": "#3498db",
                "negative": "#e74c3c",
                "unknown": "#95a5a6"
            },
            "n_examples": 3
        }
    }
    config_path = Path("config.yaml")
    if not config_path.exists():
        with open(config_path, 'w') as config_file:
            yaml.dump(default_config, config_file, default_flow_style=False)
        logger.info(f"Created default configuration file at {config_path}")

def load_components_from_config(config_path):
    """Load pipeline components from the configuration file"""
    config = Config(config_path)  # Instantiate the Config class
    processor = DataProcessor(config)
    extractor = PolicyExtractor(config)
    clusterer = PolicyClusterer(config)
    visualizer = Visualizer(config)
    return processor, extractor, clusterer, visualizer

def main():
    parser = argparse.ArgumentParser(description="Run the policy extraction and clustering pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Input file with abstracts (CSV or JSON)")
    parser.add_argument("--skip-to", choices=['load', 'extract', 'classify', 'cluster', 'visualize'],
    help="Skip to a specific pipeline stage using checkpoints")
    args = parser.parse_args()

    # Ensure the configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Initialize and run the pipeline using the implementation from policy_pipeline.py
    pipeline = PolicyPipeline(args.config, logger=logger)
    pipeline.run(args.input, args.skip_to)

if __name__ == "__main__":
    logger = setup_logging()
    create_default_config()
    main()