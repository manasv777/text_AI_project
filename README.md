# Sentiment Analysis with TF-IDF vs Sentence Embeddings

An interactive NLP project exploring sentiment analysis using traditional and modern machine learning approaches.

This repository includes both research code and a **complete Streamlit web app** that lets readers experiment with models, tune hyperparameters, and understand how different feature representations impact performance.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models Implemented](#models-implemented)
4. [Results Summary](#results-summary)
5. [Key Concepts](#key-concepts)
6. [Streamlit App Guide](#streamlit-app-guide)
7. [Installation & Setup](#installation--setup)
8. [How to Run the App](#how-to-run-the-app)
9. [App Features](#app-features)
10. [Configuration Guide](#configuration-guide)
11. [Example Workflows](#example-workflows)
12. [Troubleshooting](#troubleshooting)
13. [Architecture](#architecture)
14. [Technologies Used](#technologies-used)

---

## Project Overview

This repository explores sentiment analysis through the lens of **feature representation**:

- **Traditional Approach**: TF-IDF (term frequency-inverse document frequency) extracts interpretable text features
- **Modern Approach**: Sentence embeddings use pretrained neural models to capture semantic meaning
- **Regularization Study**: Compares L1 (sparse) and L2 (dense) regularization strategies
- **Hyperparameter Tuning**: Investigates how the C parameter (inverse regularization strength) affects model performance
- **Evaluation**: Uses 5-fold cross-validation to assess model stability and accuracy

The key finding: **Feature representation matters more than model complexity**. Embeddings significantly outperform TF-IDF on small datasets.

### Interactive Streamlit App

A full-featured Streamlit web app was built to make this project accessible to readers:

- Experiment with different models and configurations
- Enter custom reviews and get instant predictions
- Run cross-validation experiments
- Compare baseline model performance
- Explore semantic similarity using cosine distance
- No coding required—just click, configure, and learn

---

## Dataset

A curated dataset of 24 movie reviews was created for this project.

### Review Types

**Positive Reviews (12 examples)**
- Strong positives: "Amazing movie with great acting"
- Mild positives: "It was okay, not bad"

**Negative Reviews (12 examples)**
- Strong negatives: "Terrible acting and boring plot"
- Mixed cases: "Great cast but the movie was terrible"

### Labels

- `1` → Positive sentiment
- `0` → Negative sentiment

### Data Embedding

The dataset is embedded directly in the project (in `data_loader.py`), so no external files are needed.

---

## Models Implemented

### 1. TF-IDF + Logistic Regression

**Feature Representation:**
- Unigrams (single words) and bigrams (word pairs)
- Stopword removal (remove common words like "the", "a")
- Sublinear TF scaling (log-based term frequency weighting)

**Why TF-IDF?**
- Interpretable: you can see which words drive predictions
- Fast: works well for simple text classification
- Baseline: traditional NLP approach for comparison

**Regularization Options:**

| Type | Behavior | Use Case |
|------|----------|----------|
| L1 | Sparse weights; zeroes out less important features | Maximum interpretability |
| L2 | Dense weights; shrinks all coefficients | Better numerical stability |

### 2. Sentence Embeddings + Logistic Regression

**Feature Representation:**
- Pretrained model: `all-MiniLM-L6-v2` from SentenceTransformers
- Output: 384-dimensional dense vectors
- Captures semantic meaning beyond surface-level word matching

**Why Embeddings?**
- Semantic understanding: recognizes similar meanings ("movie" ≈ "film")
- Small-dataset friendly: works well with limited labeled data
- Transfer learning: uses knowledge from billions of sentences

**Regularization:**
- L2 only (standard for embeddings)
- Produces dense, stable coefficients

---

## Results Summary

### Cross-Validation Performance (5-fold CV)

| Model | Features | Regularization | Best C | Mean Accuracy | Std Dev | Notes |
|-------|----------|-----------------|--------|---------------|---------|-------|
| **TF-IDF** | Sparse text features | L1 | 0.001 | 0.50 | 0.09 | Underfitting; strong regularization removes all useful features |
| **TF-IDF** | Sparse text features | L2 | 100 | 0.49 | 0.23 | Overfitting; high variance, unstable performance |
| **Embeddings** | Dense semantic vectors | L2 | 1 | **0.96** | **0.08** | Best performance; strong semantic understanding with stability |

### Key Insight

Embeddings outperform TF-IDF by **47 percentage points** (0.96 vs 0.49 accuracy) on this small dataset. The dense semantic representation captures meaning that sparse word counts miss.

---

## Cosine Similarity Analysis

### Understanding Semantic Relationships

Cosine similarity measures how semantically related two texts are, using their embedding vectors.

**Important:** Cosine similarity ≠ sentiment agreement

**Example:**
- "Amazing movie with great acting" (positive)
- "Terrible acting and boring plot" (negative)

Result: High cosine similarity (~0.54) because both discuss movie acting, despite opposite sentiment.

### Key Takeaway

Semantic similarity and sentiment are independent dimensions. Two texts can be about the same topic but express different sentiments.

---

## Key Concepts

### C (Hyperparameter: Inverse Regularization Strength)

Controls the tradeoff between fitting training data and keeping weights small.

| C Value | Effect | Risk |
|---------|--------|------|
| **0.001–0.01** | Very strong regularization | Underfitting (too simple) |
| **0.1–1** | Moderate regularization | Balanced |
| **10–100** | Weak regularization | Overfitting (too complex) |

**How to choose:** Start with C=1; increase if underfitting, decrease if overfitting.

### L1 vs L2 Regularization

**L1 Regularization (Lasso)**
- Produces sparse solutions; some weights become exactly 0
- Automatically performs feature selection
- More interpretable (fewer active features)
- Works well for high-dimensional data

**L2 Regularization (Ridge)**
- Produces dense solutions; all weights stay active
- Shrinks large weights without eliminating them
- More numerically stable
- Better for correlated features

### Cross-Validation

Evaluates model performance by:
1. Splitting data into K folds (typically 5)
2. Training on K-1 folds, testing on 1 fold
3. Repeating K times, rotating the test fold
4. Reporting mean and standard deviation

**Why?** Prevents overfitting to a single train/test split.

### TF-IDF (Term Frequency-Inverse Document Frequency)

Traditional NLP feature extraction:

- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare a word is across all documents
- **TF-IDF Score**: TF × IDF = importance weight for each word

**Advantage:** Interpretable—you see which exact words the model uses.

### Sentence Embeddings

Pretrained neural models that convert text to dense vectors:

- **Input:** Text string
- **Process:** Neural network encoding
- **Output:** 384-dimensional vector
- **Property:** Similar texts have similar vectors

**Advantage:** Captures semantic meaning beyond word matching.

### Cosine Similarity

Measures the angle between two vectors (0 to 1 scale):

- **1.0** = identical direction (semantically identical)
- **0.5** = moderate angle (somewhat related)
- **0.0** = orthogonal (completely unrelated)

**Formula:** similarity = (vector1 · vector2) / (||vector1|| × ||vector2||)

---

## Streamlit App Guide

### What is Streamlit?

Streamlit is a Python framework for building interactive web apps with minimal code. No HTML/CSS/JavaScript required.

### What Can You Do?

The Sentiment Analysis NLP Lab app lets you:

1. **Predict sentiment** on any custom review
2. **Run experiments** with different models and hyperparameters
3. **Compare models** side-by-side
4. **Explore similarity** between texts
5. **Understand features** that drive predictions
6. **Learn NLP concepts** through interactive demos

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Navigate to Project Directory

```bash
cd /Users/manasvellaturi/Documents/text_AI_project
```

### Step 2: Activate Virtual Environment

```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements_app.txt
```

This installs:
- `streamlit` — web app framework
- `scikit-learn` — machine learning library
- `sentence-transformers` — embedding models
- `pandas` — data manipulation
- `numpy` — numerical computing
- `torch` — deep learning backend

**First-time note:** Installation may take 2–5 minutes. Subsequent runs are much faster.

### Step 4: Verify Installation

```bash
python -c "import streamlit, sklearn, sentence_transformers; print('✓ All packages installed')"
```

---

## How to Run the App

### Quick Start (3 commands)

```bash
source .venv/bin/activate
cd /Users/manasvellaturi/Documents/text_AI_project
streamlit run app.py
```

### What Happens Next

1. **Streamlit server starts** (takes 2–5 seconds)
2. **Browser opens automatically** at `http://localhost:8501`
3. **App loads** with sidebar and tabs
4. **You're ready to experiment!**

### If Browser Doesn't Open

Manually navigate to: `http://localhost:8501`

### Stop the App

Press `Ctrl+C` in the terminal where you ran `streamlit run app.py`.

---

## App Features

### Tab 1: Predict

**Purpose:** Get sentiment predictions on custom movie reviews.

**How to Use:**

1. **Configure the model** in the left sidebar:
   - Choose TF-IDF or Embeddings
   - Select regularization type (L1/L2 for TF-IDF)
   - Adjust C value with the slider
2. **Enter a review** in the text area:
   - Example: "This movie was absolutely amazing!"
3. **Click "Predict Sentiment"**
4. **See results:**
   - Predicted label (Positive / Negative)
   - Confidence score (0–100%)
   - For TF-IDF: Expand "Model Interpretation" to see which words influenced the prediction
   - For Embeddings: Note that the 384 dimensions capture semantic patterns

**Example Predictions to Try:**

- "Amazing story with brilliant acting"
- "Waste of time, terrible movie"
- "Not bad, but could have been better"
- "Best film I've ever seen"

---

### Tab 2: Cross-Validation

**Purpose:** Evaluate model stability and performance using 5-fold cross-validation.

**How to Use:**

1. **Configure the model** in the sidebar (same as Predict tab)
2. **Click "Run Cross-Validation"**
3. **Wait** (10–60 seconds depending on model):
   - TF-IDF: 5–15 seconds
   - Embeddings: 15–60 seconds (first run downloads model; subsequent runs are faster)
4. **Review results:**
   - Fold accuracies: accuracy for each of the 5 folds
   - Mean accuracy: average across all folds
   - Std dev: how stable the model is (lower is better)
   - Feature analysis (TF-IDF only): which words the model learned to use

**What You'll See:**

```
Mean Accuracy: 0.9000
Std Dev: 0.1000

Fold Results:
Fold 1: 0.8333
Fold 2: 0.9167
Fold 3: 0.8333
Fold 4: 0.9167
Fold 5: 1.0000

Top 5 Positive Features (TF-IDF only):
  "great" : 1.2345
  "amazing" : 1.1234
  "brilliant" : 0.9876
  ...

Top 5 Negative Features (TF-IDF only):
  "terrible" : -1.5432
  "awful" : -1.4321
  "boring" : -1.2345
  ...
```

**Interpretation:**

- High mean accuracy = model works well
- Low std dev = stable, consistent performance
- High std dev = performance varies across folds (may need different C value)

---

### Tab 3: Comparison

**Purpose:** Compare all three baseline model configurations side-by-side.

**How to Use:**

1. **Click "Generate Comparison Table"**
2. **Wait** (30–120 seconds) while the app evaluates:
   - TF-IDF + L1
   - TF-IDF + L2
   - Embeddings + L2
3. **Review the table:**
   - Best C value for each config
   - Mean accuracy
   - Standard deviation
   - Notes on performance

**What You'll Learn:**

- Which approach (TF-IDF vs Embeddings) performs best
- How L1 vs L2 regularization affects results
- Which C value works best for each config
- Trade-offs between interpretability and accuracy

---

### Tab 4: Cosine Similarity Explorer

**Purpose:** Understand semantic relationships between texts using embeddings.

**How to Use:**

1. **Enter two texts:**
   - Text 1 (left box): e.g., "Great movie with amazing acting"
   - Text 2 (right box): e.g., "Terrible movie with bad acting"
2. **Click "Compute Similarity"**
3. **See results:**
   - Cosine similarity score (0.0 to 1.0)
   - Interpretation: High/Moderate/Low/Distinct
   - Note about semantic vs sentiment similarity

**Example Comparisons to Try:**

| Text A | Text B | Expected Similarity | Why |
|--------|--------|-------------------|-----|
| "Amazing movie" | "Terrible movie" | HIGH | Both about movies |
| "Great acting" | "Brilliant performance" | HIGH | Semantic synonyms |
| "I loved the plot" | "I hated the plot" | MODERATE | Same topic, opposite sentiment |
| "Movie theater" | "Pizza restaurant" | LOW | Different topics |

**Key Insight:**

Cosine similarity measures topic/semantic closeness, **not** sentiment agreement. Two texts about the same subject can be very similar semantically even if they express opposite opinions.

---

## Configuration Guide

### Sidebar Controls

The left sidebar contains all model configuration options. Changes apply immediately to the next action.

### Model Type Selection

**TF-IDF + Logistic Regression**
- **Features:** Interpretable word/bigram weights
- **Speed:** Fast (5–15 sec for CV)
- **Best for:** Understanding which words matter
- **Trade-off:** Lower accuracy on small datasets

**Sentence Embeddings + Logistic Regression**
- **Features:** Dense semantic vectors (384-dimensional)
- **Speed:** Slower (15–60 sec for CV, first time downloads model)
- **Best for:** Best accuracy on small/medium datasets
- **Trade-off:** Features not directly interpretable

### Regularization Type (TF-IDF Only)

**L1 (Lasso)**
- Produces sparse weights
- Some features zeroed out
- More interpretable
- Good for high-dimensional data

**L2 (Ridge)**
- Produces dense weights
- All features active
- More stable numerically
- Better for correlated features

Embeddings use L2 by default.

### C Value Slider

Range: 0.001 to 100

**Effect:**
- **0.001–0.01:** Very strong regularization → simple model → risk of underfitting
- **0.1–1:** Moderate regularization → balanced
- **10–100:** Weak regularization → complex model → risk of overfitting

**How to Choose:**

1. Start with C=1
2. If accuracy is too low, increase C (try 5 or 10)
3. If accuracy is good but std dev is high, decrease C (try 0.1 or 0.5)
4. Experiment and watch the metrics change

### Advanced TF-IDF Options

Expand "Advanced Options" to tune:

**Stop Words On/Off**
- **On (default):** Remove common words like "the", "a", "is"
- **Off:** Keep all words
- **Recommendation:** Keep ON for better accuracy

**Sublinear TF Scaling On/Off**
- **On (default):** Use log-based term frequency (reduces weight of very frequent words)
- **Off:** Use raw term frequency
- **Recommendation:** Keep ON for better performance

**N-gram Range**

| Option | Meaning | Example |
|--------|---------|---------|
| (1,1) | Unigrams only | "amazing", "movie" |
| (1,2) | Unigrams + bigrams | "amazing", "movie", "amazing movie" |

- **Recommendation:** Use (1,2) for better feature capture

---

## Example Workflows

### Workflow 1: Quick Sentiment Check (2 minutes)

Goal: Quickly test if the app works and predict sentiment on one review.

**Steps:**

1. Keep default settings in sidebar
2. Go to "Predict" tab
3. Enter: "This movie was fantastic!"
4. Click "Predict Sentiment"
5. See: "Positive" with confidence ~0.95

---

### Workflow 2: Compare TF-IDF vs Embeddings (10 minutes)

Goal: Understand which approach works better.

**Steps:**

1. Open "Comparison" tab
2. Click "Generate Comparison Table"
3. Wait ~60 seconds
4. Review table:
   - Which has highest mean accuracy? (Embeddings, typically ~0.96)
   - Which has lowest std dev? (Embeddings, more stable)
   - How much better is embeddings? (Usually 40+ percentage points)

**Learning:** Modern embeddings far outperform traditional TF-IDF on small datasets.

---

### Workflow 3: Experiment with C Values (15 minutes)

Goal: See how regularization affects model performance.

**Steps:**

1. Select "TF-IDF + Logistic Regression"
2. Select L2 regularization
3. Set C = 0.001
4. Go to "Cross-Validation" tab
5. Click "Run Cross-Validation"
6. Record mean accuracy and std dev
7. Increase C to 1, repeat step 5
8. Increase C to 100, repeat step 5
9. Compare results:
   - Which C gave best accuracy?
   - Which C gave lowest std dev?
   - What's the trade-off?

**Learning:** There's a sweet spot for C that balances accuracy and stability.

---

### Workflow 4: Understand Feature Importance (20 minutes)

Goal: See which words TF-IDF models use for predictions.

**Steps:**

1. Select "TF-IDF + Logistic Regression"
2. Select L1 regularization
3. Set C = 1
4. Go to "Cross-Validation" tab
5. Click "Run Cross-Validation"
6. Expand "Top Positive Features"
7. Note the words (e.g., "amazing", "great", "excellent")
8. Review "Top Negative Features" (e.g., "terrible", "awful", "boring")
9. Change C to 100 and repeat
10. Compare: Do the important features stay the same? Which C gives clearer features?

**Learning:** Feature importance shifts with regularization. L1 produces sparser, more interpretable models.

---

### Workflow 5: Explore Semantic Similarity (10 minutes)

Goal: Understand how embeddings capture meaning.

**Steps:**

1. Go to "Cosine Similarity Explorer" tab
2. Compare three pairs:
   - "Amazing movie" vs "Terrible movie" → HIGH (both about movies)
   - "Great acting" vs "Brilliant performance" → HIGH (synonyms)
   - "I love pizza" vs "I love movies" → MODERATE (same verb, different object)
3. For each, note the similarity score
4. Click example buttons to see more comparisons

**Learning:** Embeddings capture topic and semantic relationships, separate from sentiment.

---

## Troubleshooting

### Issue: ModuleNotFoundError when running app

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**

1. Confirm virtual environment is active: `source .venv/bin/activate`
2. Reinstall dependencies: `pip install -r requirements_app.txt`
3. Try again: `streamlit run app.py`

---

### Issue: App takes a long time to start

**Cause:** First run downloads the embedding model (~100 MB).

**Solution:**

- Be patient (1–2 minutes on first run)
- Subsequent runs are much faster (5–10 seconds)
- Internet connection required for first run only

---

### Issue: All predictions are the same class (always "Positive" or "Negative")

**Cause:** Model is underfitting; regularization is too strong.

**Solution:**

1. In sidebar, increase C value (try 5 or 10)
2. If using L1, try switching to L2
3. Run prediction again

---

### Issue: High standard deviation in cross-validation

**Cause:** Model is overfitting or data is inconsistent.

**Solution:**

1. In sidebar, decrease C value (try 0.1 or 0.5)
2. For TF-IDF, try L1 for more regularization
3. Run cross-validation again

---

### Issue: "Enter a review first" warning

**Cause:** You clicked "Predict" without typing anything.

**Solution:** Type or paste a review before clicking predict.

---

### Issue: Streamlit app won't open in browser

**Cause:** Server started but browser didn't open automatically.

**Solution:** Manually navigate to `http://localhost:8501`

---

### Issue: App crashes or shows errors

**Solution:**

1. Check the terminal where you ran `streamlit run app.py`
2. Look for error messages
3. Screenshot and note the error
4. Common fixes:
   - Restart the app: Ctrl+C and `streamlit run app.py`
   - Reinstall packages: `pip install -r requirements_app.txt`

---

## Architecture

### File Structure

```
/Users/manasvellaturi/Documents/text_AI_project/
├── app.py                         # Main Streamlit UI
├── data_loader.py                 # Load reviews and labels
├── tfidf_wrapper.py               # TF-IDF functions
├── embedding_wrapper.py           # Embedding functions
├── evaluation_wrapper.py          # Comparison utilities
├── requirements_app.txt           # Dependencies
├── README.md                      # This file
├── APP_USAGE.md                   # Detailed app guide
├── QUICKSTART.md                  # Quick reference
├── IMPLEMENTATION_SUMMARY.md      # Technical details
├── DELIVERY_SUMMARY.md            # Delivery notes
├── SETUP_CHECKLIST.md             # Setup verification
│
├── compare_models.py              # Original research code
├── embeddings_model.py            # Original research code
├── train_model.py                 # Original research code
├── l1vsl2.py                      # Original research code
├── crossval_experiment.py         # Original research code
└── ... (other original files)
```

### How the App Works

```
User clicks button in Streamlit UI
    ↓
app.py receives interaction
    ↓
Calls wrapper function (tfidf_wrapper.py, embedding_wrapper.py, etc.)
    ↓
Wrapper trains model using scikit-learn or sentence-transformers
    ↓
Model makes predictions or runs cross-validation
    ↓
Results sent back to Streamlit
    ↓
UI displays results to user
```

### Original Code Preservation

All original research scripts are **100% untouched**:
- `compare_models.py`
- `embeddings_model.py`
- `train_model.py`
- `l1vsl2.py`
- `crossval_experiment.py`
- `ctest.py`
- `embeddings_test.py`
- `baseline_l1_model.py`
- `model_selection.py`
- `learn.py`

The wrapper modules replicate logic without importing or modifying originals.

---

## Technologies Used

| Technology | Purpose | Version |
|-----------|---------|---------|
| Python | Programming language | 3.8+ |
| Streamlit | Web UI framework | 1.28+ |
| scikit-learn | Machine learning library | 1.3+ |
| sentence-transformers | Embedding models | 2.2+ |
| pandas | Data manipulation | 2.0+ |
| numpy | Numerical computing | 1.24+ |
| torch | Deep learning backend | 2.0+ |

---

## Quick Reference Commands

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements_app.txt
```

### Run the App

```bash
streamlit run app.py
```

### Stop the App

```
Ctrl+C (in terminal)
```

### Check Package Versions

```bash
pip list | grep -E 'streamlit|scikit-learn|sentence-transformers'
```

---

## Notes for Substack Readers

This repository is designed to make sentiment analysis interactive and accessible.

### Getting Started (5 minutes)

1. Install dependencies (see Installation & Setup section)
2. Run the app: `streamlit run app.py`
3. Try the "Predict" tab with a movie review
4. Play with the sidebar controls
5. Explore other tabs

### Understanding the Project (30 minutes)

1. Read the "Key Concepts" section
2. Run the "Comparison" tab to see TF-IDF vs Embeddings
3. Try the "Cross-Validation" tab with different C values
4. Explore "Cosine Similarity" to understand embeddings

### Deep Dive (1+ hour)

1. Review the "Results Summary" and "Architecture" sections
2. Read `APP_USAGE.md` for detailed feature explanations
3. Study the original code (preserved in `compare_models.py`, `embeddings_model.py`, etc.)
4. Experiment with different configurations in the app

---

## Final Notes

This project demonstrates that **feature representation is more important than model complexity**. Even a simple logistic regression model can achieve 96% accuracy when using the right features (embeddings).

The Streamlit app makes this lesson interactive. Experiment, learn, and explore!

**Ready to start?**

```bash
source .venv/bin/activate
cd /Users/manasvellaturi/Documents/text_AI_project
streamlit run app.py
```

Open your browser at `http://localhost:8501` and enjoy exploring!
