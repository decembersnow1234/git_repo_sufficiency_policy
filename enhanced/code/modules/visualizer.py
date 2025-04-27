import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, count
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
from modules.config import Config

class Visualizer:
    """Visualize policy clusters and impacts using PySpark & Spark NLP"""

    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.output_dir = Path(config.output_dir)

    def visualize_clusters(self, policy_df, cluster_df):
        """Generate visualizations for policy clusters using Spark"""
        if policy_df.count() == 0 or cluster_df.count() == 0:
            logging.warning("No policies or clusters to visualize")
            return

        # Create output directory if not exists
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Generate visualizations
        self._plot_impact_distribution(cluster_df, viz_dir)
        self._generate_cluster_report(policy_df, cluster_df, viz_dir)

    def _plot_impact_distribution(self, cluster_df, viz_dir: Path):
        """Plot impact distribution using Spark DataFrame"""
        impact_distribution = cluster_df.groupBy("cluster").agg(
            count("text").alias("total_policies"),
            count(when(col("impact") == "positive", True)).alias("positive"),
            count(when(col("impact") == "neutral", True)).alias("neutral"),
            count(when(col("impact") == "negative", True)).alias("negative"),
            count(when(col("impact") == "unknown", True)).alias("unknown"),
        ).toPandas()  # Convert to Pandas for visualization
        
        # Ensure data is available
        if impact_distribution.empty:
            logging.warning("No impact data to visualize")
            return

        # Plot stacked bars efficiently
        fig, ax = plt.subplots(figsize=(12, 8))
        impact_distribution.set_index("cluster").plot(kind="bar", stacked=True, ax=ax, color=['#2ecc71', '#3498db', '#e74c3c', '#95a5a6'])

        # Customize plot
        ax.set_title('Policy Impact Distribution by Cluster', fontsize=16)
        ax.set_ylabel('Number of Policies', fontsize=14)
        ax.set_xlabel('Cluster', fontsize=14)
        ax.legend(title="Impact Type")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(viz_dir / "impact_distribution.png", dpi=300)
        plt.close()

    def _generate_cluster_report(self, policy_df, cluster_df, viz_dir: Path):
        """Generate cluster report using Spark DataFrame operations"""
        report_sections = ["# Policy Cluster Analysis Report\n\n"]
        
        cluster_data = cluster_df.collect()
        for row in cluster_data:
            impact_distribution = f"- Positive: {row['positive']}\n- Neutral: {row['neutral']}\n- Negative: {row['negative']}\n- Unknown: {row['unknown']}\n"
            
            representative_policies = policy_df.filter(col("cluster") == row["cluster"]) \
                                               .select("text").limit(5).collect()
            policy_list = "\n".join(f"{i}. {p.text}" for i, p in enumerate(representative_policies, 1))

            report_sections.append(
                f"## Cluster {row['cluster']}\n\n"
                f"**Size:** {row['total_policies']} policies\n\n"
                f"**Impact Distribution:**\n{impact_distribution}\n\n"
                f"**Representative Policies:**\n{policy_list}\n\n"
                "---\n"
            )

        # Write report to file
        with open(viz_dir / "cluster_report.md", "w") as f:
            f.write("\n".join(report_sections))

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = Config("config.yaml")
    spark = SparkSession.builder \
        .appName("PolicyVisualizer") \
        .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:4.4.0") \
        .getOrCreate()

    visualizer = Visualizer(config, spark)

    # Load policy and cluster data (example)
    policy_df = spark.read.parquet("processed_policies.parquet")
    cluster_df = spark.read.parquet("cluster_metadata.parquet")

    visualizer.visualize_clusters(policy_df, cluster_df)
