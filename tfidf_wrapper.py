import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_tfidf_model(reviews, labels, C, penalty='l2', ngram_range=(1, 2), 
                      stop_words='english', sublinear_tf=True):
    """
    Train a TF-IDF + Logistic Regression model and return both model and vectorizer.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        sublinear_tf=sublinear_tf,
    )
    X = vectorizer.fit_transform(reviews)
    model = LogisticRegression(
        penalty=penalty,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        C=C
    )
    model.fit(X, labels)
    return model, vectorizer, X


def predict_with_tfidf(model, vectorizer, text):
    """
    Predict sentiment for a single review using TF-IDF model.
    Returns label (0 or 1) and probability.
    """
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = probabilities[prediction]
    return prediction, confidence


def evaluate_tfidf_cv(reviews, labels, C, penalty='l2', ngram_range=(1, 2),
                      stop_words='english', sublinear_tf=True, cv=5):
    """
    Run 5-fold cross-validation for TF-IDF model.
    Returns fold scores, mean accuracy, and std dev.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        sublinear_tf=sublinear_tf,
    )
    X = vectorizer.fit_transform(reviews)
    model = LogisticRegression(
        penalty=penalty,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        C=C
    )
    scores = cross_val_score(model, X, labels, cv=cv)
    return scores, scores.mean(), scores.std()


def get_tfidf_top_features(model, vectorizer, top_n=5):
    """
    Extract top positive and negative features from a trained TF-IDF model.
    Returns two lists: (top_positive, top_negative) with (feature_name, coefficient) tuples.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    top_positive_indices = coefficients.argsort()[-top_n:][::-1]
    top_negative_indices = coefficients.argsort()[:top_n]
    
    top_positive = [(feature_names[idx], coefficients[idx]) for idx in top_positive_indices]
    top_negative = [(feature_names[idx], coefficients[idx]) for idx in top_negative_indices]
    
    return top_positive, top_negative


def get_tfidf_nonzero_features_count(model, vectorizer):
    """
    For L1 regularization, return number of non-zero features.
    """
    coefficients = model.coef_[0]
    return int(np.sum(coefficients != 0))
