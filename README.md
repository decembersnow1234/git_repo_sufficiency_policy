# Codes created for the article: "Evidence-based classification of sufficiency policies for mobility using Machine Learning on 18,000 articles."
It contains 6 Jupyter Notebooks for the extraction and analysis of sufficiency policies from a screened dataset of articles:
- 1_policy_extraction
- 2_policy_clustering
- 3_outcomes_clustering
- 4_measure_consensus
- 5_heatmap
- 6_sufficiency_index


## 1_policy_extraction - Extraction of policies, outcomes and correlations

This code allows the extraction of policies, their outcomes and the correlations between them.

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

This code allows the culstering of policies.

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
- The dataset of papers with the policy extraction of the 1_policy_extraction code. 

At the end of this script you will extract: 
- The named_cluster_df dataset of policy clusters and the sentences extracted during the 1_policy_extraction. 


## 3_outcomes_clustering - Clustering process of the policies, cleaning of the correlations and preparing of the data analysis. 

This code allows the culstering of outcomes and the cleaning of the correlations and preparing of the data analysis.

It uses Sentence BERT as an embedder and HDBSCAN as a clustering algorithm. 
- Sentence BERT is a Python module for accessing, using, and training state-of-the-art text and image embedding models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder models.
- HDBSCAN is a clustering algorithm extending DBSCAN by converting it into a hierarchical clustering algorithm. DBSCAN is a density-based clustering method that finds core samples and expands clusters from them. 

In this Jupyter Notebook we will: 
1. Import the data retrieved from the policy extraction process ; 
2. Import the relevant packages ;
3. Prepare data for clustering ;
    1. Filtering ;
4. Cluster with HDBSCAN ; 
5. Re-process the clusters ; 
    1. Export for manual check ;
    2. Reclustering with HDBSCAN ;
    3. Clean and name the final clusters ;
6. Meta clustering. 

To complete those tasks you will need:
- The dataset of papers with the policy extraction of the 1_policy_extraction code. 
- The dataset of papers with the clustered policy of the 2_policy_clustering code. 

At the end of this script you will extract: 
- The named_cluster_df dataset of outcome clusters and the sentences extracted during the 1_policy_extraction. 
- The prepared dataset for the data analysis. 


## 4_similarity_score - Computing of prevalence and building of the consensus matrix.

This code allows the calculation of similarity of policy and abstract.

It uses the outputs of the other phases that will be used to compute the matrix of correlations pondered by a similarity score. The similarity score is cmputed using Proportional Sentence Match.

In this Jupyter Notebook we will: 
1. Import the data retrieved from the policy and outcome clustering process ; 
2. Import the relevant packages ;
3. Prepare data for computing ;
4. Prevalence of policies in abstract Using Proportional Sentence Match ; 
5. Export data with prevalence measure.

To complete those tasks you will need:
- The dataset of papers with the policy extraction of the 1_policy_extraction code. 
- The dataset of papers with the clustered policy of the 2_policy_clustering code. 
- The dataset of papers with the clustered policy of the 3_outcomes_clustering code. 

At the end of this script you will extract: 
- The named_cluster_df dataset of policies with prevalence metrics. 


## 5_heatmap - Computing the heatmap matrix 

This code allows the calculation of the heatmap of consensus on sense of correlation between policies and outcomes.

It uses the outputs of the other phases that will be used to compute the heatmap matrix of sum of correlations pondered by the similarity score. 

In this Jupyter Notebook we will: 
1. Import the data with similarity score ; 
2. Import the relevant packages ;
3. Prepare data for computing ;
4. Compute the heatmap ; 
5. Export heatmap data.

To complete those tasks you will need:
- The dataset of papers with the policy extraction of the 4_similarity_score code. 

At the end of this script you will extract: 
- The heatmap_df dataset of sum of correlations pondered by the similarity score. 


## 6_sufficiency_index - Computing sufficiency indices and comparing policies

This code allows the calculation of the sufficiency index to compare policies extracted.

It uses the outputs of the other phases that will be used to draw a bubble graph for each policies classified by Lower Limit and Upper Limit scores. The scores are calculated as geometric means of non-null correlation values. 

In this Jupyter Notebook we will: 
1. Import the data with similarity score ; 
2. Import the relevant packages ;
3. Sufficiency index calculus ;
4. Drawing bubble graph ; 
5. Export bubble graph data.
    1. Select per quantile

To complete those tasks you will need:
- The dataset of papers with the policy extraction of the 4_similarity_score code. 

At the end of this script you will extract: 
- The heatmap_df dataset of sum of correlations pondered by the similarity score. 