import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import yaml
from tqdm import tqdm
from collections import Counter

from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

from modules.config import Config

class PolicyClusterer:
    """Cluster policy statements based on semantic similarity and impact"""

    def __init__(self, config: Config):
        self.config = config
        model_name = config.get_param('models', 'impact_classifier', 'distilbert-base-uncased')
        self.impact_classifier = pipeline("text-classification", model=model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), device=0 if torch.cuda.is_available() else -1)

    def classify_impact(self, policies: List[Dict]) -> List[Dict]:
        """Classify policy impacts as positive, neutral, or negative."""
        impact_mapping = {
            'positive': ['improve', 'increase', 'enhance', 'benefit'],
            'negative': ['reduce', 'decrease', 'limit', 'restrict']
        }

        for policy in tqdm(policies, desc="Classifying policy impacts"):
            try:
                text = policy['text'].lower()
                policy['impact'] = next((impact for impact, words in impact_mapping.items() if any(word in text for word in words)), 'neutral')
                policy['impact_confidence'] = 0.8  # Placeholder confidence score
            except Exception as e:
                logging.warning(f"Impact classification error: {e}")
                policy['impact'], policy['impact_confidence'] = 'unknown', 0.0

        return policies

    def cluster_policies(self, policies: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Cluster policy statements based on semantic similarity."""
        if not policies:
            logging.warning("No policies to cluster")
            return policies, {}

        embeddings = np.array([policy['embedding'] for policy in policies])
        best_n_clusters = self._find_optimal_clusters(embeddings, min(20, len(policies) // 5) if len(policies) > 10 else 2)

        clusters = AgglomerativeClustering(n_clusters=best_n_clusters, linkage='ward', metric='euclidean').fit_predict(embeddings)

        for i, policy in enumerate(policies):
            policy['cluster_id'] = int(clusters[i])

        cluster_meta = self._generate_cluster_metadata(policies, best_n_clusters)

        logging.info(f"Clustered policies into {best_n_clusters} groups")
        return policies, cluster_meta

    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int) -> int:
        """Find optimal number of clusters using silhouette score."""
        silhouette_scores = [(n_clusters, silhouette_score(embeddings, KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)))
                             for n_clusters in range(2, max_clusters + 1) if n_clusters < len(embeddings)]

        return max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else min(3, max_clusters)

    def _generate_cluster_metadata(self, policies: List[Dict], n_clusters: int) -> Dict:
        """Generate metadata for each cluster."""
        cluster_meta = {}

        for cluster_id in range(n_clusters):
            cluster_policies = [p for p in policies if p['cluster_id'] == cluster_id]
            if not cluster_policies:
                continue

            impact_counts = Counter(p.get('impact', 'unknown') for p in cluster_policies)

            embeddings = np.array([p['embedding'] for p in cluster_policies])
            centroid = embeddings.mean(axis=0)
            representative_policies = [cluster_policies[idx]['text'] for idx in np.argsort([np.linalg.norm(emb - centroid) for emb in embeddings])[:3]]

            cluster_meta[cluster_id] = {
                'size': len(cluster_policies),
                'impact_distribution': dict(impact_counts),
                'primary_impact': max(impact_counts, key=impact_counts.get),
                'representative_policies': representative_policies,
                'common_actors': self._extract_common_elements([p.get('actors', []) for p in cluster_policies]),
            }

        return cluster_meta

    def _extract_common_elements(self, list_of_lists: List[List]) -> List:
        """Extract common elements appearing in multiple lists."""
        flattened_counts = Counter(element for sublist in list_of_lists for element in sublist)
        threshold = max(2, len(list_of_lists) // 5)

        return [elem for elem, count in sorted(flattened_counts.items(), key=lambda x: x[1], reverse=True) if count >= threshold][:5]
