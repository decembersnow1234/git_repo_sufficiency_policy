#!/usr/bin/env python
# run_pipeline.py
"""
Script to run the policy extraction and clustering pipeline.
"""

import argparse
import logging
from pathlib import Path
import yaml
import sys
import time
from policy_pipeline import PolicyPipeline

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