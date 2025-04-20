import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from modules.config import Config

class Visualizer:
    """Visualize policy clusters and impacts"""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.output_dir

    def visualize_clusters(self, policies: List[Dict], cluster_meta: Dict):
        """Generate visualizations for policy clusters"""
        if not policies:
            logging.warning("No policies to visualize")
            return

        # Create directories for outputs
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Generate visualizations and reports
        self._plot_impact_distribution(cluster_meta, viz_dir)
        self._generate_cluster_report(policies, cluster_meta, viz_dir)

    def _plot_impact_distribution(self, cluster_meta: Dict, viz_dir: Path):
        """Plot impact distribution by cluster"""
        clusters = list(cluster_meta.keys())
        if not clusters:
            logging.warning("No clusters to visualize")
            return

        # Extract impact distributions
        impact_types = ['positive', 'neutral', 'negative', 'unknown']
        impact_data = {impact: [cluster_meta[c]['impact_distribution'].get(impact, 0) for c in clusters] for impact in impact_types}

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.6
        bottom_values = np.zeros(len(clusters))

        # Plot stacked bars efficiently
        colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c', 'unknown': '#95a5a6'}
        for impact in impact_types:
            ax.bar(clusters, impact_data[impact], width, bottom=bottom_values, label=impact.capitalize(), color=colors[impact])
            bottom_values += np.array(impact_data[impact])

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
        report_sections = [f"# Policy Cluster Analysis Report\n\nAnalysis completed with {len(policies)} total policy statements\n"]

        for cluster_id, meta in sorted(cluster_meta.items()):
            impact_distribution = "\n".join([
                f"- {impact.capitalize()}: {count} ({(count / meta['size'] * 100):.1f}%)"
                for impact, count in meta['impact_distribution'].items() if meta['size'] > 0
            ])

            common_actors = "\n".join(f"- {actor}" for actor in meta.get('common_actors', []))
            representative_policies = "\n".join(f"{i}. {policy}" for i, policy in enumerate(meta['representative_policies'], 1))

            report_sections.append(
                f"## Cluster {cluster_id}\n\n"
                f"**Size:** {meta['size']} policies\n\n"
                f"**Primary Impact:** {meta['primary_impact'].capitalize()}\n\n"
                f"**Impact Distribution:**\n{impact_distribution}\n\n"
                f"{'**Common Policy Actors:**\n' + common_actors + '\n\n' if common_actors else ''}"
                f"**Representative Policy Statements:**\n{representative_policies}\n\n"
                "---\n"
            )

        # Write report to file
        with open(viz_dir / "cluster_report.md", "w") as f:
            f.write("\n".join(report_sections))
