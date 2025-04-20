import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm
from collections import Counter

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import yaml

from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer

from modules.config import Config

class PolicyExtractor:
    """Extract policy statements from scientific abstracts using transformer models"""

    def __init__(self, config: Config):
        self.config = config
        model_name = config.get_param('models', 'policy_extractor', 'roberta-base')
        
        # Load models - in production, these would be fine-tuned models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentence_classifier = pipeline("text-classification", model=model_name, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.sentence_encoder = SentenceTransformer('all-mpnet-base-v2')

        # Load policy keywords
        self.policy_keywords = set(config.get_param('extraction', 'policy_keywords', [
            'policy', 'regulation', 'legislation', 'law', 'rule', 'guideline',
            'framework', 'program', 'initiative', 'strategy', 'plan', 'measure',
            'intervention', 'approach', 'mechanism', 'instrument', 'proposal'
        ]))

    def extract_policies(self, processed_docs: List[Dict]) -> List[Dict]:
        """Extract policy statements from processed documents using multiple methods"""
        policy_statements = [
            self._extract_policy_details(sentence, doc)
            for doc in tqdm(processed_docs, desc="Extracting policies")
            for sentence in doc['sentences'] if len(sentence.split()) >= 5 and self._is_policy_statement(sentence, doc)
        ]

        logging.info(f"Extracted {len(policy_statements)} policy statements")
        return policy_statements

    def _is_policy_statement(self, sentence: str, doc: Dict) -> bool:
        """Determine if a sentence contains a policy statement using multiple signals"""
        sentence_lower = sentence.lower()
        
        keyword_match = any(keyword in sentence_lower for keyword in self.policy_keywords)
        transformer_confidence = 0.5  # Placeholder confidence score
        
        try:
            classification = self.sentence_classifier(sentence)
            if classification and classification[0]['label'] == 'policy':
                transformer_confidence = classification[0]['score']
        except Exception as e:
            logging.warning(f"Classifier error: {e}")

        syntax_pattern = any(
            token.get('dep') == 'aux' and any(t.get('dep') == 'ROOT' and t.get('pos') == 'VERB' for t in doc['tokens'])
            for token in doc['tokens'] if isinstance(token, dict) and token.get('text', '').lower() in {'should', 'must', 'need', 'could', 'would'}
        )

        return (keyword_match and transformer_confidence > 0.3) or (syntax_pattern and keyword_match)

    def _extract_policy_details(self, sentence: str, doc: Dict) -> Dict:
        """Extract structured information from a policy statement"""
        embedding = self.sentence_encoder.encode(sentence)

        policy = {
            'text': sentence,
            'embedding': embedding,
            'actors': [ent['text'] for ent in doc['entities'] if ent['label'] in {'ORG', 'GPE', 'PERSON'} and ent['text'] in sentence],
            'targets': [],  # Placeholder for NLP-based extraction
            'mechanisms': [],  # Placeholder for NLP-based extraction
            'source_doc': f"{doc['text'][:100]}..."  # Reference snippet
        }

        return policy
