# policy_pipeline.py
"""
Centralized pipeline for policy extraction and clustering from scientific abstracts
with enhanced NLP models and semantic clustering.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
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


class DataProcessor:
    """Data loading and preprocessing for scientific abstracts"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        
    def load_data(self, input_file: str) -> pd.DataFrame:
        """Load data from CSV, JSON, or Excel file"""
        file_path = self.config.data_dir / input_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    
    def preprocess_abstracts(self, abstracts: List[str]) -> List[Dict]:
        """
        Preprocess abstracts with spaCy for enhanced NLP analysis
        Returns list of processed documents with annotations
        """
        processed_docs = []
        
        for abstract in tqdm(abstracts, desc="Preprocessing abstracts"):
            if not isinstance(abstract, str) or not abstract.strip():
                continue
                
            doc = self.nlp(abstract)
            
            # Extract key information
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
                        'text': token, 
                        'pos': token.pos_,
                        'dep': token.dep_
                    }
                    for token in doc if not token.is_stop and not token.is_punct
                ]
            }
            processed_docs.append(processed_doc)
        
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs


class PolicyExtractor:
    """Extract policy statements from scientific abstracts using transformer models"""
    
    def __init__(self, config: Config):
        self.config = config
        model_name = config.get_param('models', 'policy_extractor', 'roberta-base')
        
        # Load models - in production would use specific fine-tuned models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For policy sentence classification
        self.sentence_classifier = pipeline(
            "text-classification", 
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # For policy entity extraction
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Sentence embedding model for semantic analysis
        self.sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Load policy keywords from config or use defaults
        self.policy_keywords = config.get_param(
            'extraction', 
            'policy_keywords', 
            ['policy', 'regulation', 'legislation', 'law', 'rule', 'guideline',
             'framework', 'program', 'initiative', 'strategy', 'plan', 'measure',
             'intervention', 'approach', 'mechanism', 'instrument', 'proposal']
        )
    
    def extract_policies(self, processed_docs: List[Dict]) -> List[Dict]:
        """
        Extract policy statements from processed documents using a multi-method approach
        """
        policy_statements = []
        
        for doc in tqdm(processed_docs, desc="Extracting policies"):
            # Extract at sentence level
            for sentence in doc['sentences']:
                # Skip short sentences
                if len(sentence.split()) < 5:
                    continue
                    
                # Check if this is likely a policy statement using multiple signals
                is_policy = self._is_policy_statement(sentence, doc)
                
                if is_policy:
                    # Extract policy details - actors, targets, mechanisms
                    policy = self._extract_policy_details(sentence, doc)
                    policy_statements.append(policy)
        
        logger.info(f"Extracted {len(policy_statements)} policy statements")
        return policy_statements
    
    def _is_policy_statement(self, sentence: str, doc: Dict) -> bool:
        """
        Determine if a sentence contains a policy statement using multiple signals
        """
        # 1. Check for policy keywords
        keyword_match = any(keyword in sentence.lower() for keyword in self.policy_keywords)
        
        # 2. Use transformer classifier for policy statement identification
        try:
            classification = self.sentence_classifier(sentence)
            # Note: This would be replaced with a fine-tuned policy classifier
            # This is a placeholder that would need a custom trained model
            transformer_confidence = 0.5  # Placeholder for demo purposes
        except Exception as e:
            logger.warning(f"Classifier error: {e}")
            transformer_confidence = 0
        
        # 3. Check syntax patterns typical of policy statements
        # Example: Modal verbs + action verbs
        syntax_pattern = any(token.get('dep') == 'aux' and 
                        any(t.get('dep') == 'ROOT' and t.get('pos') == 'VERB' 
                            for t in doc['tokens']) 
                        for token in doc['tokens'] 
                        if token.get('text').lower() in ['should', 'must', 'need', 'could', 'would'])
        
        # Combine signals (would be tuned based on experiments)
        is_policy = (keyword_match and transformer_confidence > 0.3) or (syntax_pattern and keyword_match)
        
        return is_policy
    
    def _extract_policy_details(self, sentence: str, doc: Dict) -> Dict:
        """
        Extract structured information from a policy statement
        """
        # Placeholder for demonstration - would use NER and dependency parsing
        # to extract actors, targets, and policy mechanisms
        
        # Get sentence embedding for later clustering
        embedding = self.sentence_encoder.encode(sentence)
        
        policy = {
            'text': sentence,
            'embedding': embedding,
            'actors': [ent['text'] for ent in doc['entities'] 
                      if ent['label'] in ['ORG', 'GPE', 'PERSON'] 
                      and ent['text'] in sentence],
            'targets': [],  # Would be populated with advanced NLP
            'mechanisms': [],  # Would be populated with advanced NLP
            'source_doc': doc['text'][:100] + '...',  # Reference to source
        }
        
        return policy


class PolicyClusterer:
    """Cluster policy statements based on semantic similarity and impact"""
    
    def __init__(self, config: Config):
        self.config = config
        self.impact_classifier = pipeline(
            "text-classification", 
            model=config.get_param('models', 'impact_classifier', 'distilbert-base-uncased'),
            tokenizer=AutoTokenizer.from_pretrained(
                config.get_param('models', 'impact_classifier', 'distilbert-base-uncased')
            ),
            device=0 if torch.cuda.is_available() else -1
        )
    
    def classify_impact(self, policies: List[Dict]) -> List[Dict]:
        """
        Classify policy impacts as positive, neutral, or negative
        """
        for policy in tqdm(policies, desc="Classifying policy impacts"):
            # In production, this would use a fine-tuned model on policy impact data
            # This is a simplified placeholder
            try:
                # Placeholder logic - would be replaced with actual model predictions
                text = policy['text'].lower()
                if any(word in text for word in ['improve', 'increase', 'enhance', 'benefit']):
                    impact = 'positive'
                elif any(word in text for word in ['reduce', 'decrease', 'limit', 'restrict']):
                    impact = 'negative'
                else:
                    impact = 'neutral'
                
                policy['impact'] = impact
                policy['impact_confidence'] = 0.8  # Placeholder confidence score
            except Exception as e:
                logger.warning(f"Impact classification error: {e}")
                policy['impact'] = 'unknown'
                policy['impact_confidence'] = 0.0
        
        return policies
    
    def cluster_policies(self, policies: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Cluster policy statements based on semantic similarity
        """
        if not policies:
            logger.warning("No policies to cluster")
            return policies, {}
        
        # Extract embeddings
        embeddings = np.array([policy['embedding'] for policy in policies])
        
        # Find optimal number of clusters
        max_clusters = min(20, len(policies) // 5) if len(policies) > 10 else 2
        best_n_clusters = self._find_optimal_clusters(embeddings, max_clusters)
        
        # Perform hierarchical clustering
        cluster_model = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            affinity='euclidean',
            linkage='ward'
        )
        clusters = cluster_model.fit_predict(embeddings)
        
        # Assign cluster IDs to policies
        for i, policy in enumerate(policies):
            policy['cluster_id'] = int(clusters[i])
        
        # Generate cluster metadata
        cluster_meta = self._generate_cluster_metadata(policies, best_n_clusters)
        
        logger.info(f"Clustered policies into {best_n_clusters} groups")
        return policies, cluster_meta
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int) -> int:
        """
        Find optimal number of clusters using silhouette score
        """
        silhouette_scores = []
        
        # Try different cluster counts
        for n_clusters in range(2, max_clusters + 1):
            # Skip if we have too few samples
            if n_clusters >= len(embeddings):
                continue
                
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append((n_clusters, silhouette_avg))
            except Exception as e:
                logger.warning(f"Error calculating silhouette for k={n_clusters}: {e}")
        
        if not silhouette_scores:
            return min(3, max_clusters)
        
        # Find cluster count with best score
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        return best_n_clusters
    
    def _generate_cluster_metadata(self, policies: List[Dict], n_clusters: int) -> Dict:
        """
        Generate metadata for each cluster
        """
        cluster_meta = {}
        
        for cluster_id in range(n_clusters):
            cluster_policies = [p for p in policies if p['cluster_id'] == cluster_id]
            
            # Skip empty clusters
            if not cluster_policies:
                continue
                
            # Calculate impact distribution
            impact_counts = {
                'positive': sum(1 for p in cluster_policies if p.get('impact') == 'positive'),
                'neutral': sum(1 for p in cluster_policies if p.get('impact') == 'neutral'),
                'negative': sum(1 for p in cluster_policies if p.get('impact') == 'negative'),
                'unknown': sum(1 for p in cluster_policies if p.get('impact') == 'unknown')
            }
            
            # Find representative policies (closest to cluster centroid)
            embeddings = np.array([p['embedding'] for p in cluster_policies])
            centroid = embeddings.mean(axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
            representative_idx = np.argsort(distances)[:3]
            representative_policies = [cluster_policies[idx]['text'] for idx in representative_idx]
            
            # Create metadata
            cluster_meta[cluster_id] = {
                'size': len(cluster_policies),
                'impact_distribution': impact_counts,
                'primary_impact': max(impact_counts, key=impact_counts.get),
                'representative_policies': representative_policies,
                'common_actors': self._extract_common_elements([p.get('actors', []) for p in cluster_policies]),
            }
        
        return cluster_meta
    
    def _extract_common_elements(self, list_of_lists: List[List]) -> List:
        """Extract common elements that appear in multiple lists"""
        if not list_of_lists:
            return []
            
        # Flatten and count occurrences
        all_elements = []
        for sublist in list_of_lists:
            all_elements.extend(sublist)
            
        # Count occurrences
        element_counts = {}
        for element in all_elements:
            if element in element_counts:
                element_counts[element] += 1
            else:
                element_counts[element] = 1
        
        # Return elements that appear more than once, sorted by frequency
        threshold = max(2, len(list_of_lists) // 5)  # Appear in at least 20% of policies
        common = [(elem, count) for elem, count in element_counts.items() if count >= threshold]
        common.sort(key=lambda x: x[1], reverse=True)
        
        return [elem for elem, _ in common[:5]]  # Return top 5 common elements


class Visualizer:
    """Visualize policy clusters and impacts"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.output_dir
    
    def visualize_clusters(self, policies: List[Dict], cluster_meta: Dict):
        """Generate visualizations for policy clusters"""
        if not policies:
            logger.warning("No policies to visualize")
            return
            
        # Create directories for outputs
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Plot impact distribution by cluster
        self._plot_impact_distribution(cluster_meta, viz_dir)
        
        # 2. Generate cluster summary report
        self._generate_cluster_report(policies, cluster_meta, viz_dir)
    
    def _plot_impact_distribution(self, cluster_meta: Dict, viz_dir: Path):
        """Plot impact distribution by cluster"""
        # Prepare data
        clusters = []
        positive = []
        neutral = []
        negative = []
        unknown = []
        
        for cluster_id, meta in cluster_meta.items():
            clusters.append(f"Cluster {cluster_id}")
            impacts = meta['impact_distribution']
            positive.append(impacts['positive'])
            neutral.append(impacts['neutral'])
            negative.append(impacts['negative'])
            unknown.append(impacts['unknown'])
        
        if not clusters:
            logger.warning("No clusters to visualize")
            return
            
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.6
        
        # Plot stacked bars
        ax.bar(clusters, positive, width, label='Positive', color='#2ecc71')
        ax.bar(clusters, neutral, width, bottom=positive, label='Neutral', color='#3498db')
        ax.bar(clusters, negative, width, bottom=[p+n for p,n in zip(positive, neutral)], 
               label='Negative', color='#e74c3c')
        ax.bar(clusters, unknown, width, 
               bottom=[p+n+ng for p,n,ng in zip(positive, neutral, negative)], 
               label='Unknown', color='#95a5a6')
        
        # Customize plot
        ax.set_title('Policy Impact Distribution by Cluster', fontsize=16)
        ax.set_ylabel('Number of Policies', fontsize=14)
        ax.set_xlabel('Cluster', fontsize=14)
        ax.legend()
        
        # Save plot
        fig.tight_layout()
        plt.savefig(viz_dir / "impact_distribution.png", dpi=300)
        plt.close()
    
    def _generate_cluster_report(self, policies: List[Dict], cluster_meta: Dict, viz_dir: Path):
        """Generate detailed cluster report"""
        report = "# Policy Cluster Analysis Report\n\n"
        report += f"Analysis completed with {len(policies)} total policy statements\n\n"
        
        # For each cluster
        for cluster_id, meta in sorted(cluster_meta.items()):
            report += f"## Cluster {cluster_id}\n\n"
            report += f"**Size:** {meta['size']} policies\n\n"
            report += f"**Primary Impact:** {meta['primary_impact'].capitalize()}\n\n"
            
            report += "**Impact Distribution:**\n"
            for impact, count in meta['impact_distribution'].items():
                percentage = (count / meta['size']) * 100 if meta['size'] > 0 else 0
                report += f"- {impact.capitalize()}: {count} ({percentage:.1f}%)\n"
            report += "\n"
            
            if meta.get('common_actors'):
                report += "**Common Policy Actors:**\n"
                for actor in meta['common_actors']:
                    report += f"- {actor}\n"
                report += "\n"
            
            report += "**Representative Policy Statements:**\n"
            for i, policy in enumerate(meta['representative_policies'], 1):
                report += f"{i}. {policy}\n"
            report += "\n"
            
            report += "---\n\n"
        
        # Write report to file
        with open(viz_dir / "cluster_report.md", "w") as f:
            f.write(report)


class PolicyPipeline:
    """Main pipeline orchestrating the policy extraction and clustering process"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline with configuration"""
        self.config = Config(config_path)
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
        
        # Convert numpy arrays to lists for JSON serialization
        if stage == 'extraction' or stage == 'classification':
            for item in data:
                if 'embedding' in item:
                    item['embedding'] = item['embedding'].tolist()
        elif stage == 'clustering':
            for item in data['policies']:
                if 'embedding' in item:
                    item['embedding'] = item['embedding'].tolist()
        
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved checkpoint for stage: {stage}")
    
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