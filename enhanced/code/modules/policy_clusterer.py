import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
from pyspark.ml.clustering import KMeans
from collections import Counter
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
        impact_udf = udf(self._get_impact_label, StringType())
        confidence_udf = udf(lambda _: 0.8, FloatType())  # Placeholder confidence

        df = df.withColumn("impact", impact_udf(df.text))
        df = df.withColumn("impact_confidence", confidence_udf(df.text))

        logging.info("Policy impact classification completed")
        return df

    def _get_impact_label(self, text: str) -> str:
        """Determine impact based on keywords"""
        text_lower = text.lower()
        for impact, words in self.impact_mapping.items():
            if any(word in text_lower for word in words):
                return impact
        return "neutral"

    def cluster_policies(self, df):
        """Use Spark MLlib KMeans clustering on policy embeddings"""
        kmeans = KMeans(featuresCol="embedding", k=10)
        model = kmeans.fit(df)
        df = model.transform(df)

        logging.info("Policy clustering completed")
        return df

    def generate_cluster_metadata(self, df):
        """Generate cluster metadata using Spark operations"""
        cluster_summary = df.groupBy("prediction").agg(
            col("text").alias("sample_text"),
            col("impact").alias("primary_impact"),
            col("impact_confidence").alias("avg_confidence"),
        ).orderBy(col("avg_confidence").desc())

        logging.info("Cluster metadata generated")
        return cluster_summary
