import logging
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from transformers import AutoTokenizer, pipeline
from modules.config import Config

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

        # Ensure the embedding column exists
        if "embedding" not in df.columns:
            raise ValueError("DataFrame does not contain column 'embedding' for clustering")

        # Convert embeddings to Spark-compatible format
        assembler = VectorAssembler(inputCols=["embedding"], outputCol="features")
        df = assembler.transform(df)

        # Convert Spark DataFrame column into NumPy array
        embeddings = np.array(df.select("embedding").toPandas()["embedding"].tolist())

        if len(embeddings) < 2:
            raise ValueError("Not enough embeddings to perform clustering")

        max_clusters = min(20, len(embeddings) // 5) if len(embeddings) > 10 else 2

        # Fit KMeans clustering
        kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=max_clusters)
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

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = Config("config.yaml")
    spark = SparkSession.builder.appName("PolicyClusterer").getOrCreate()

    clusterer = PolicyClusterer(config, spark)

    input_file = "processed_data.parquet"
    df = spark.read.parquet(input_file)
    df = clusterer.classify_impact(df)
    df = clusterer.cluster_policies(df)
    cluster_summary = clusterer.generate_cluster_metadata(df)

    cluster_summary.show()
