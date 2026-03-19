# Sentiment Analysis with TF-IDF vs Sentence Embeddings

This project explores different approaches to sentiment analysis, starting from traditional text features (TF-IDF) and progressing to modern NLP techniques using pretrained sentence embeddings.

The goal is to understand how feature representation impacts model performance.

---

## Project Overview

In this project, I built and evaluated multiple sentiment classification models using both classical and modern NLP approaches.

I started with TF-IDF features and logistic regression, then transitioned to pretrained sentence embeddings to understand how feature representation impacts performance.

---

## Dataset

I created a small dataset of movie reviews including:

- Positive reviews (e.g., *"Amazing movie with great acting"*)
- Negative reviews (e.g., *"Terrible acting and boring plot"*)
- Mixed/edge cases (e.g., *"Great acting but terrible plot"*)

Labels:
- `1` → Positive  
- `0` → Negative  

---

## Models Implemented

### 1. TF-IDF + Logistic Regression
- Features:
  - Unigrams and bigrams (`ngram_range=(1,2)`)
  - Stopword removal
  - Sublinear term frequency scaling
- Regularization:
  - L1 (sparse model)
  - L2 (dense model)

---

### 2. Sentence Embeddings + Logistic Regression
- Model: `all-MiniLM-L6-v2` (SentenceTransformers)
- Produces dense 384-dimensional vectors
- Captures semantic meaning instead of just word frequency

---

## Training & Evaluation

- Used 5-fold cross-validation
- Tuned hyperparameter C (regularization strength)
- Evaluated:
  - Mean accuracy
  - Standard deviation (stability)

---

## Results

| Model | Feature Representation | Regularization | Best C | Mean CV Accuracy | Std Dev | Notes |
|---|---|---|---|---:|---:|---|
| Logistic Regression | TF-IDF (1,2)-grams + stopwords removed + sublinear TF | L1 | 0.001 | 0.50 | 0.09 | Strong regularization → no learned features (underfitting) |
| Logistic Regression | TF-IDF (1,2)-grams + stopwords removed + sublinear TF | L2 | 100 | 0.49 | 0.23 | High variance, unstable performance |
| Logistic Regression | Sentence Embeddings (MiniLM) | L2 | 1 | 0.96 | 0.08 | Best performance, strong semantic understanding |

---

## Cosine Similarity Analysis

I computed cosine similarity between sentence embeddings to understand semantic relationships.

Key observation:

Cosine similarity measures semantic similarity, not sentiment.

Example:
- "Amazing movie with great acting"
- "Terrible acting and boring plot"

These sentences have moderate similarity because they discuss the same topic (movie acting), even though their sentiment is opposite.

---

## Key Insights

### 1. Feature representation matters more than model complexity
- TF-IDF struggles on small datasets
- Embeddings significantly improve performance

---

### 2. Regularization must be tuned carefully
- L1 with very small C → removes all features (underfitting)
- L2 with large C → leads to unstable models (overfitting)

---

### 3. Embeddings capture meaning, not just words
- Recognize relationships like:
  - "movie" ≈ "film"
  - "amazing" ≈ "fantastic"
- Enable semantic comparison using cosine similarity

---

### 4. Classification vs similarity
- Cosine similarity → measures semantic closeness
- Logistic regression → learns sentiment boundaries

---

## Conclusion

This project demonstrates that pretrained sentence embeddings significantly outperform traditional TF-IDF features for sentiment analysis, especially on small datasets.

---

## Technologies Used

- Python
- scikit-learn
- sentence-transformers
- NumPy

---

## How to Run

```bash
pip install scikit-learn sentence-transformers numpy
python embeddings_model.py