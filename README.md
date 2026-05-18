# TextScope — Sentiment Analysis: TF-IDF vs Sentence Embeddings

An NLP project comparing traditional and modern approaches to sentiment analysis, with an interactive Streamlit app for hands-on model exploration.

## Overview

This project investigates how **feature representation** affects sentiment classification performance. Using a curated movie review dataset, it benchmarks two approaches:

- **TF-IDF + Logistic Regression** — interpretable, traditional NLP
- **Sentence Embeddings + Logistic Regression** — semantic, modern NLP

Key finding: **embeddings outperform TF-IDF by ~47 percentage points** (0.96 vs 0.49 accuracy) on small datasets, demonstrating that feature representation matters more than model complexity.

## Features

- End-to-end ML pipeline: data loading → feature extraction → model training → evaluation
- Two feature representations: sparse TF-IDF and dense sentence embeddings (`all-MiniLM-L6-v2`)
- Regularization comparison: L1 (Lasso) vs L2 (Ridge) with hyperparameter tuning
- 5-fold cross-validation for robust performance estimation
- Cosine similarity explorer for semantic relationship analysis
- Interactive Streamlit web app — no coding required to experiment

## Results

| Model | Features | Regularization | Mean Accuracy | Std Dev |
|-------|----------|---------------|---------------|---------|
| TF-IDF | Sparse text features | L1 | 0.50 | 0.09 |
| TF-IDF | Sparse text features | L2 | 0.49 | 0.23 |
| **Embeddings** | Dense semantic vectors | **L2** | **0.96** | **0.08** |

## Tech Stack

- **Python** — core language
- **scikit-learn** — logistic regression, TF-IDF, cross-validation
- **sentence-transformers** — `all-MiniLM-L6-v2` embedding model
- **Streamlit** — interactive web app
- **pandas / numpy** — data processing

## Project Structure
text_AI_project/
├── app.py                  # Streamlit frontend
├── data_loader.py          # Dataset (embedded, no external files needed)
├── tfidf_wrapper.py        # TF-IDF feature extraction + model training
├── embedding_wrapper.py    # Sentence embedding model
├── evaluation_wrapper.py   # Cross-validation and comparison utilities
├── compare_models.py       # Model benchmarking scripts
├── crossval_experiment.py  # Cross-validation experiments
├── l1vsl2.py               # Regularization comparison
├── requirements_app.txt    # Dependencies
└── README.md

## Getting Started

```bash
git clone https://github.com/manasv777/text_AI_project.git
cd text_AI_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_app.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## App Features

- **Predict** — enter any movie review and get a sentiment prediction with confidence score
- **Cross-Validation** — run 5-fold CV and inspect fold-by-fold accuracy
- **Comparison** — benchmark all three model configs side-by-side
- **Cosine Similarity Explorer** — compare any two texts semantically using embeddings

## Key Concepts Demonstrated

- TF-IDF feature extraction with unigrams/bigrams and sublinear scaling
- L1 vs L2 regularization trade-offs in high-dimensional text classification
- Transfer learning via pretrained sentence transformers
- Cross-validation methodology and interpretation
- Cosine similarity as a measure of semantic relatedness (distinct from sentiment)
