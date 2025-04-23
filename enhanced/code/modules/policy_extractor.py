import logging
import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline, LightPipeline
from sparknlp.annotator import (
    Tokenizer, 
    ClassifierDLModel, 
    NerDLModel, 
    NerConverter
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType
from modules.config import Config

class PolicyExtractor:
    """Extract policy statements using Spark NLP & PySpark"""

    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark

        # NLP preprocessing pipeline
        self.pipeline = self._build_spark_nlp_pipeline()

    def _build_spark_nlp_pipeline(self):
        """Define NLP preprocessing pipeline using Spark NLP"""
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("tokens")

        classifier = ClassifierDLModel.pretrained("classifierdl_bertwikiner_policy", "en") \
            .setInputCols(["document"]) \
            .setOutputCol("policy_classification")

        ner_model = NerDLModel.pretrained("onto_100", "en") \
            .setInputCols(["document", "tokens"]) \
            .setOutputCol("ner")

        ner_converter = NerConverter() \
            .setInputCols(["document", "ner"]) \
            .setOutputCol("entities")

        pipeline = Pipeline(stages=[
            document_assembler, tokenizer, classifier, ner_model, ner_converter
        ])

        return pipeline.fit(self.spark.createDataFrame([[""]], ["text"]))  # Fit empty DF to initialize models

    def extract_policies(self, df):
        """Extract policy statements using Spark NLP"""
        processed_df = self.pipeline.transform(df) \
            .select(
                "text", 
                col("policy_classification.result").alias("policy_confidence"), 
                col("entities.result").alias("entities")
            )

        logging.info(f"Extracted {processed_df.count()} policy statements")
        return processed_df

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = Config("config.yaml")
    spark = SparkSession.builder \
        .appName("PolicyExtractor") \
        .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:4.4.0") \
        .getOrCreate()

    extractor = PolicyExtractor(config, spark)

    input_file = "processed_data.parquet"
    df = spark.read.parquet(input_file)

    processed_df = extractor.extract_policies(df)

    # Schema for saving extracted policies
    schema = StructType([
        StructField("text", StringType(), True),
        StructField("policy_confidence", ArrayType(StringType()), True),
        StructField("entities", ArrayType(StringType()), True)
    ])

    policy_df = spark.createDataFrame(processed_df.collect(), schema)
    policy_df.show()
