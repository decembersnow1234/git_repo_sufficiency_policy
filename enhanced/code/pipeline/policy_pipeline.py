# policy_pipeline.py
"""
Centralized pipeline for policy extraction and clustering from scientific abstracts
with enhanced NLP models and semantic clustering.
"""
from policy_extractor import PolicyExtractor
from policy_clusterer import PolicyClusterer
from data_processor import DataProcessor
from nlp_encoder import NLPEncoder

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    BertModel
)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("policy_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set default paths
        self.data_dir = Path(self.config.get('paths', {}).get('data_dir', 'data'))
        self.output_dir = Path(self.config.get('paths', {}).get('output_dir', 'output'))
        self.model_dir = Path(self.config.get('paths', {}).get('model_dir', 'models'))
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Set model configurations
        self.models = self.config.get('models', {})
        
        # Processing parameters
        self.params = self.config.get('parameters', {})
        
    def get_param(self, section: str, param: str, default=None):
        """Get a specific parameter with fallback to default"""
        return self.config.get(section, {}).get(param, default)




class PolicyPipeline:
    """Main pipeline orchestrating the policy extraction and clustering process"""
    
    def __init__(self, config_path: str = "config.yaml", logger=None):
        """Initialize the pipeline with configuration"""
        self.config = Config(config_path)
        self.logger = logger
        self.processor = DataProcessor(self.config)
        self.extractor = PolicyExtractor(self.config)
        self.clusterer = PolicyClusterer(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Create checkpoint directory
        self.checkpoint_dir = self.config.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def run(self, input_file: str, skip_to: Optional[str] = None):
        """Run the full pipeline with optional checkpoint resumption"""
        logger.info(f"Starting policy extraction pipeline on {input_file}")
        
        if skip_to is None or skip_to == 'load':
            # Load and preprocess data
            df = self.processor.load_data(input_file)
            abstracts = df['abstract'].tolist() if 'abstract' in df.columns else df['text'].tolist()
            processed_docs = self.processor.preprocess_abstracts(abstracts)
            self._save_checkpoint('preprocessing', processed_docs)
        else:
            processed_docs = self._load_checkpoint('preprocessing')
        
        if skip_to is None or skip_to in ['load', 'extract']:
            # Extract policies
            policies = self.extractor.extract_policies(processed_docs)
            self._save_checkpoint('extraction', policies)
        else:
            policies = self._load_checkpoint('extraction')
        
        if skip_to is None or skip_to in ['load', 'extract', 'classify']:
            # Classify policy impacts
            policies = self.clusterer.classify_impact(policies)
            self._save_checkpoint('classification', policies)
        else:
            policies = self._load_checkpoint('classification')
        
        if skip_to is None or skip_to in ['load', 'extract', 'classify', 'cluster']:
            # Cluster policies
            policies, cluster_meta = self.clusterer.cluster_policies(policies)
            self._save_checkpoint('clustering', {'policies': policies, 'cluster_meta': cluster_meta})
        else:
            checkpoint = self._load_checkpoint('clustering')
            policies = checkpoint['policies']
            cluster_meta = checkpoint['cluster_meta']
        
        # Visualize results
        self.visualizer.visualize_clusters(policies, cluster_meta)
        
        # Save final results
        self._save_final_output(policies, cluster_meta)
        
        logger.info(f"Pipeline completed successfully. Results saved to {self.config.output_dir}")
        return policies, cluster_meta
    
    def _save_checkpoint(self, stage: str, data):
        """Save checkpoint data to disk"""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"

        # Create a deep copy to avoid modifying the original data
        data_copy = copy.deepcopy(data)
        
        try:
            # Use the custom encoder to handle all serialization
            with open(checkpoint_file, 'w') as f:
                json.dump(data_copy, f, cls=NLPEncoder)
            logger.info(f"Saved checkpoint for stage: {stage}")
        except TypeError as e:
            logger.error(f"Serialization error in {stage} checkpoint: {e}")
            # Attempt more aggressive object conversion if normal serialization fails
            if stage == 'preprocessing':
                # Convert any remaining problematic objects to strings
                serializable_data = self._make_fully_serializable(data_copy)
                with open(checkpoint_file, 'w') as f:
                    json.dump(serializable_data, f)
                logger.info(f"Saved {stage} checkpoint after additional processing")
    
    def _make_fully_serializable(self, data):
        """Recursively convert any non-serializable objects to serializable forms"""
        if isinstance(data, dict):
            return {k: self._make_fully_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_fully_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'text') and hasattr(data, 'pos_'):  # Likely a spaCy token
            return {
                'text': str(data.text),
                'pos': data.pos_,
                'dep': data.dep_
            }
        elif hasattr(data, '__dict__') and not isinstance(data, (str, int, float, bool)):
            # Any other custom objects with attributes
            return str(data)
        else:
            # Primitive types that are already serializable
            return data
    
    def _load_checkpoint(self, stage: str):
        """Load checkpoint data from disk"""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        if stage == 'extraction' or stage == 'classification':
            for item in data:
                if 'embedding' in item:
                    item['embedding'] = np.array(item['embedding'])
        elif stage == 'clustering':
            for item in data['policies']:
                if 'embedding' in item:
                    item['embedding'] = np.array(item['embedding'])
        
        logger.info(f"Loaded checkpoint for stage: {stage}")
        return data
    
    def _save_final_output(self, policies: List[Dict], cluster_meta: Dict):
        """Save final processed data"""
        # Convert numpy arrays for JSON serialization
        serializable_policies = []
        for policy in policies:
            policy_copy = policy.copy()
            if 'embedding' in policy_copy:
                policy_copy['embedding'] = policy_copy['embedding'].tolist()
            serializable_policies.append(policy_copy)
        
        # Save policies
        with open(self.config.output_dir / "policies.json", 'w') as f:
            json.dump(serializable_policies, f, indent=2)
        
        # Save cluster metadata
        with open(self.config.output_dir / "cluster_metadata.json", 'w') as f:
            json.dump(cluster_meta, f, indent=2)
        
        # Save as CSV for easier analysis
        policy_df = pd.DataFrame([
            {
                'text': p['text'],
                'impact': p.get('impact', 'unknown'),
                'cluster_id': p.get('cluster_id', -1),
                'actors': ', '.join(p.get('actors', [])),
            }
            for p in policies
        ])
        policy_df.to_csv(self.config.output_dir / "policies.csv", index=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and cluster policy statements from scientific abstracts")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Input file with abstracts (CSV or JSON)")
    parser.add_argument("--skip-to", choices=['load', 'extract', 'classify', 'cluster', 'visualize'], 
                        help="Skip to a specific pipeline stage using checkpoints")
    
    args = parser.parse_args()
    
    pipeline = PolicyPipeline(args.config)
    pipeline.run(args.input, args.skip_to)