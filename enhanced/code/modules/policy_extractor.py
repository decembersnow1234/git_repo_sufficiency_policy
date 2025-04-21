import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import spacy
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from modules.config import Config
import torch

class PolicyExtractor:
    """Extract policy statements from scientific abstracts using transformer models & PySpark"""

    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        model_name = config.get_param('models', 'policy_extractor', 'roberta-base')

        # Load models - fine-tuned for policy extraction
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

    def extract_policies(self, df):
        """Extract policies using Spark UDF"""
        policy_udf = udf(self._extract_policy_details, ArrayType(StringType()))

        # Ensure correct column reference using col("abstract")
        df = df.withColumn("policy_statements", policy_udf(col("abstract")))

        logging.info("Policy extraction completed in parallel using Spark UDF")
        return df

    def _extract_policy_details(self, text: str) -> list:
        """Extract structured information from a policy statement"""

        if not text or not isinstance(text, str):  # Handle cases where text is None or not valid
            return []

        # Load SpaCy inside function to avoid Spark UDF serialization issues
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        # Extract policy sentences
        policy_sentences = [sent.text for sent in doc.sents if any(word in sent.text.lower() for word in self.policy_keywords)]

        # Perform entity extraction
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents if ent.label_ in {'ORG', 'GPE', 'PERSON'}]

        # Generate embeddings for clustering
        embeddings = [self.sentence_encoder.encode(sent) for sent in policy_sentences]

        return [{"text": sent, "entities": entities, "embedding": emb.tolist()} for sent, emb in zip(policy_sentences, embeddings)]
