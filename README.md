# Codes created for the article: "Evidence-based classification of sufficiency policies for mobility using Machine Learning on 18,000 articles."
It contains 6 Jupyter Notebooks for the extraction and analysis of sufficiency policies from a screened dataset of articles:
- 1_policy_extraction
- 2_policy_clustering
- 3_outcomes_clustering
- 4_measure_consensus
- 5_heatmap
- 6_sufficiency_index


## 1_policy_extraction - Extraction of policies, outcomes and correlations

This code permits the extraction of policies, their outcomes and the correlations between them.

It uses gpt-4o-mini from the OpenAI library. 
gpt-4o-mini, is a compact yet high-performance Large Language Model (LLM) developed by OpenAI. While GPT-4o-mini is not open-source, and its development and training processes lack transparency due to restricted access via OpenAI’s servers, its performance has been demonstrated to surpass other models.

In this Jupyter Notebook we will: 
1. Import the data retrieved from the screening process ; 
2. Import the relevant packages ;
3. Extract policies, outcomes and correlations with gpt-4o-mini ;
4. Export the data. 

To complete those tasks you will need:
- A dataset of screened papers relevant to your research question (db_init) ; 
- A OpenAi account. 

At the end of this script you will extract: 
- The db_init dataset with a JSON format column containing the extracted policies, outcomes and correlations. 


## 2_policy_clustering - Clustering process of the policies. 

This code permits the culstering of policies.

It uses Sentence BERT as an embedder and HDBSCAN as a clustering algorithm. 
- Sentence BERT is a Python module for accessing, using, and training state-of-the-art text and image embedding models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder models.
- HDBSCAN is a clustering algorithm extending DBSCAN by converting it into a hierarchical clustering algorithm. DBSCAN is a density-based clustering method that finds core samples and expands clusters from them. 

In this Jupyter Notebook we will: 
1. Import the data retrieved from the policy extraction process ; 
2. Import the relevant packages ;
3. Prepare data for clustering ;
4. Cluster with HDBSCAN ; 
5. Re-process the clusters ; 
    1. Export for manual check ;
    2. Reclustering with HDBSCAN ;
    3. Clean and name the final clusters ;
6. Export the policy clustering data. 

To complete those tasks you will need:
- The dataset of papers with the policy extraction of 1_policy_extraction code. 

At the end of this script you will extract: 
- The named_cluster_df dataset of policy clusters and the sentences extracted during the 1_policy_extraction. 