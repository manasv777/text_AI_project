import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


def get_embedder():
    """
    Load the sentence transformer embedder.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def train_embedding_model(reviews, labels, C, penalty='l2'):
    """
    Train a Sentence Embedding + Logistic Regression model.
    Returns model and embeddings.
    """
    embedder = get_embedder()
    X = embedder.encode(reviews)
    
    model = LogisticRegression(
        penalty=penalty,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        C=C
    )
    model.fit(X, labels)
    return model, embedder, X


def predict_with_embeddings(model, embedder, text):
    """
    Predict sentiment for a single review using embeddings model.
    Returns label (0 or 1) and probability.
    """
    X = embedder.encode([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = probabilities[prediction]
    return prediction, confidence


def evaluate_embedding_cv(reviews, labels, C, penalty='l2', cv=5):
    """
    Run 5-fold cross-validation for embedding model.
    Returns fold scores, mean accuracy, and std dev.
    """
    embedder = get_embedder()
    X = embedder.encode(reviews)
    
    model = LogisticRegression(
        penalty=penalty,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        C=C
    )
    scores = cross_val_score(model, X, labels, cv=cv)
    return scores, scores.mean(), scores.std()


def compute_cosine_similarity(text1, text2):
    """
    Compute cosine similarity between two text strings using embeddings.
    """
    embedder = get_embedder()
    emb1 = embedder.encode([text1])
    emb2 = embedder.encode([text2])
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity
