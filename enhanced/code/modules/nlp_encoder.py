import json
import numpy as np
import sparknlp
from sparknlp.annotation import Annotation
from typing import Any
from pyspark.sql import SparkSession
from sparknlp.base import LightPipeline

class NLPEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling non-serializable NLP objects in Spark NLP."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Annotation):
            return {'text': obj.result, 'metadata': obj.metadata}
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if hasattr(obj, '__dict__'):
            return vars(obj)  # Safer way to serialize objects
        
        return super().default(obj)

# Example usage
if __name__ == "__main__":
    # Initialize Spark NLP
    spark = SparkSession.builder \
        .appName("NLPEncoderExample") \
        .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:4.4.0") \
        .getOrCreate()
    
    # Sample Spark NLP pipeline
    from sparknlp.base import DocumentAssembler
    from sparknlp.annotator import Tokenizer

    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("tokens")

    sample_df = spark.createDataFrame([["This is an example sentence."]], ["text"])

    pipeline = LightPipeline(document_assembler.fit(sample_df))
    result = pipeline.annotate("This is an example sentence.")

    data = {
        'tokens': result['tokens'],
        'embedding': np.array([1, 2, 3])
    }

    json_data = json.dumps(data, cls=NLPEncoder, indent=4)
    print(json_data)
