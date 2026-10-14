# AG News – Interactive Text Visualization

## Project Overview

This project is an interactive text visualization application developed using Python and Streamlit for **DATS 6401 – Visualization of Complex Data, Week 7 Assignment** at George Washington University.

The application explores news articles using word frequency analysis, topic modeling, and dimensionality reduction. It provides interactive visualizations to help users identify common vocabulary, explore underlying topics, and understand patterns in text data.

## Dataset

**Dataset:** AG News  
**Source:** [Hugging Face – AG News](https://huggingface.co/datasets/fancyzhx/ag_news)

The application uses a reproducible, balanced sample of 2,000 news articles, with 500 articles from each category:

- World
- Sports
- Business
- Sci/Tech

## Features

### 1. Dataset Overview
Displays dataset statistics, category distributions, and original news articles.

### 2. Word Frequency Visualization
Visualizes the most frequently occurring words after text preprocessing and stop-word removal. Users can select news categories and adjust the number of displayed words.

### 3. Topic Modeling
Uses Latent Dirichlet Allocation (LDA) to identify four latent topics. Includes an interactive topic selector, topic-word probabilities, topic prevalence, representative articles, and a category-topic heatmap.

### 4. Document Projection
Uses TF-IDF and Truncated Singular Value Decomposition (TruncatedSVD) to project document representations into two dimensions.

Documents can be colored by news category or dominant LDA topic. The application also reports explained variance and discusses the limitations of dimensionality reduction.

### 5. TF-IDF vs. Raw Word Counts
Compares raw word frequencies with average TF-IDF scores to illustrate how different text-weighting methods affect word rankings.

### 6. Interpretations and Limitations
Each visualization includes explanations of what the results reveal and how the visualization could be misleading.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Hugging Face Datasets

## Installation and Running

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application loads the AG News dataset from Hugging Face when a supported local dataset file is unavailable.

## Methodology

Text preprocessing includes lowercasing, HTML entity decoding, removing URLs and punctuation, and stop-word filtering.

LDA is used for exploratory topic modeling, while TF-IDF and TruncatedSVD provide a computationally lightweight two-dimensional document projection.

The visualizations are exploratory and should not be interpreted as definitive measures of semantic similarity or topic classification.

## Course Information

**Course:** DATS 6401 – Visualization of Complex Data  
**Assignment:** Week 7 – Text Data Visualization  
**University:** The George Washington University

## Dataset Reference

AG News Dataset: https://huggingface.co/datasets/fancyzhx/ag_news
