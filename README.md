
# Policy Extraction and Clustering Pipeline

This repository contains the code for extracting and clustering policy statements from a dataset of scientific abstracts. The pipeline is designed to handle large datasets and perform complex NLP tasks using advanced machine learning models.

## Overview

The pipeline consists of several stages, each responsible for a specific part of the process:

1. **Data Loading and Preprocessing**:
   - **Purpose**: Load and preprocess the input data to prepare it for policy extraction.
   - **Approach**:
     - **spaCy**: We use spaCy for its robust NLP capabilities, including tokenization, entity recognition, and dependency parsing. This helps in extracting structured information from text data efficiently.
   - **Steps**:
     - Load data from CSV, JSON, or Excel files.
     - Preprocess abstracts using spaCy to extract tokens, entities, and dependencies.
   - **Output**: Preprocessed documents with annotations.

2. **Policy Extraction**:
   - **Purpose**: Extract policy statements from the preprocessed documents.
   - **Approach**:
     - **Fine-tuned Transformer Model**: We use a transformer-based model (e.g., RoBERTa) fine-tuned on policy-related texts to accurately identify and extract policy statements. Transformer models are well-suited for this task due to their ability to capture context and semantic meaning.
   - **Steps**:
     - Use the transformer model to identify policy statements.
     - Extract key information such as actors, targets, and mechanisms.
   - **Output**: Extracted policy statements with metadata.

3. **Impact Classification**:
   - **Purpose**: Classify the impact of each policy statement as positive, neutral, or negative.
   - **Approach**:
     - **Fine-tuned Transformer Model**: We use a transformer model fine-tuned on impact classification tasks. This model can accurately classify the sentiment or impact of policy statements based on the context provided in the text.
   - **Steps**:
     - Use the fine-tuned transformer model to classify policy impacts.
     - Assign confidence scores to each classification.
   - **Output**: Classified policy statements with impact labels and confidence scores.

4. **Policy Clustering**:
   - **Purpose**: Cluster policy statements based on semantic similarity.
   - **Approach**:
     - **Sentence Transformers**: We use Sentence Transformers to generate embeddings for policy statements. These models are designed to produce semantically meaningful sentence embeddings, which are crucial for clustering similar statements together.
     - **Agglomerative Clustering**: We apply hierarchical clustering to group similar policies. Agglomerative clustering is chosen for its ability to create a hierarchy of clusters, which can be useful for understanding the structure of the data.
   - **Steps**:
     - Generate embeddings for policy statements using Sentence Transformers.
     - Apply hierarchical clustering to group similar policies based on their embeddings.
   - **Output**: Clustered policy statements with cluster metadata.

5. **Visualization**:
   - **Purpose**: Generate visualizations to analyze the clustered policy statements.
   - **Approach**:
     - **Matplotlib and Seaborn**: These libraries are used to create visualizations that help in understanding the distribution and characteristics of the clustered policies. Visualizations are essential for interpreting the results and communicating insights effectively.
   - **Steps**:
     - Create plots to visualize the impact distribution by cluster.
     - Generate a detailed cluster report for further analysis.
   - **Output**: Visualizations and reports for analyzing the clustered policy statements.

## Requirements

- Python 3.x
- Required libraries: `pandas`, `spacy`, `transformers`, `sentence-transformers`, `scikit-learn`, `matplotlib`, `seaborn`
- A dataset of scientific abstracts in CSV, JSON, or Excel format

## Usage

1. **Activate the Conda Environment**:
   ```bash
   conda activate your_env_name
   ```

2. **Run the Pipeline**:
   ```bash
   python run_pipeline.py --input path/to/your/datafile.xlsx --skip-to load
   ```

3. **Monitor Progress**:
   - Check the log files and system resources to monitor the progress of each stage.

4. **Inspect Output**:
   - Verify the output files and visualizations generated at each stage.
