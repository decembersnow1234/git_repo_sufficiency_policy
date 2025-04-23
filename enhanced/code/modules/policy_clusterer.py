import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
from pyspark.ml.clustering import KMeans
from collections import Counter
from transformers import AutoTokenizer, pipeline
from modules.config import Config
import torch

class PolicyClusterer:
    """Cluster policy statements based on semantic similarity and impact using PySpark"""

    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        model_name = config.get_param('models', 'impact_classifier', 'distilbert-base-uncased')

        # Load transformer model for impact classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.impact_classifier = pipeline("text-classification", model=model_name, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

        # Define impact mapping
        self.impact_mapping = {
            'positive': ['improve', 'increase', 'enhance', 'benefit'],
            'negative': ['reduce', 'decrease', 'limit', 'restrict']
        }
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int) -> int:
    """Find optimal number of clusters using silhouette score."""
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
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
    
    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return best_n_clusters    

    def classify_impact(self, df):
        """Classify policy impacts using Spark UDF"""

        def get_impact_label(text: str) -> str:
            """Determine impact based on keywords"""
            if not text or not isinstance(text, str):  # Handle empty values
                return "neutral"
            
            text_lower = text.lower()
            for impact, words in self.impact_mapping.items():
                if any(word in text_lower for word in words):
                    return impact
            return "neutral"

        # Convert function to Spark UDF
        impact_udf = udf(get_impact_label, StringType())
        confidence_udf = udf(lambda _: 0.8, FloatType())  # Placeholder confidence

        # Ensure correct column reference
        df = df.withColumn("impact", impact_udf(col("text")))
        df = df.withColumn("impact_confidence", confidence_udf(col("text")))

        logging.info("Policy impact classification completed")
        return df

    def cluster_policies(self, df):
        """Use Spark MLlib KMeans clustering on policy embeddings"""
        
        # Ensure the embedding column is present
        if "embedding" not in df.columns:
            raise ValueError("DataFrame does not contain column 'embedding' for clustering")

        # Convert embedding column to float type if necessary
        df = df.withColumn("embedding", col("embedding").cast(FloatType()))

        kmeans = KMeans(featuresCol="embedding", predictionCol="cluster", k=10)
        model = kmeans.fit(df)
        df = model.transform(df)

        logging.info("Policy clustering completed")
        return df

    def generate_cluster_metadata(self, df):
        """Generate cluster metadata using Spark operations"""
        cluster_summary = df.groupBy("cluster").agg(
            col("text").alias("sample_text"),
            col("impact").alias("primary_impact"),
            col("impact_confidence").alias("avg_confidence"),
        ).orderBy(col("avg_confidence").desc())

        logging.info("Cluster metadata generated")
        return cluster_summary
