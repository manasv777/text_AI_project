import pandas as pd
from data_loader import get_reviews_and_labels
from tfidf_wrapper import evaluate_tfidf_cv, get_tfidf_top_features, train_tfidf_model
from embedding_wrapper import evaluate_embedding_cv, train_embedding_model


def get_baseline_comparison_table():
    """
    Compute and return a comparison table of baseline models.
    This runs cross-validation for each model type and returns results as a DataFrame.
    """
    reviews, labels = get_reviews_and_labels()
    C_values = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    
    results = []
    
    for C in C_values:
        scores_l1, mean_l1, std_l1 = evaluate_tfidf_cv(reviews, labels, C, penalty='l1', cv=5)
        model_l1, vec_l1, _ = train_tfidf_model(reviews, labels, C, penalty='l1')
        nonzero_l1 = sum(1 for coef in model_l1.coef_[0] if coef != 0)
        results.append({
            'Model': 'TF-IDF + Logistic Regression',
            'Regularization': 'L1',
            'C': C,
            'Mean CV Accuracy': round(mean_l1, 4),
            'Std Dev': round(std_l1, 4),
            'Non-zero Features': nonzero_l1
        })
    
    for C in C_values:
        scores_l2, mean_l2, std_l2 = evaluate_tfidf_cv(reviews, labels, C, penalty='l2', cv=5)
        results.append({
            'Model': 'TF-IDF + Logistic Regression',
            'Regularization': 'L2',
            'C': C,
            'Mean CV Accuracy': round(mean_l2, 4),
            'Std Dev': round(std_l2, 4),
            'Non-zero Features': '-'
        })
    
    for C in C_values:
        scores_emb, mean_emb, std_emb = evaluate_embedding_cv(reviews, labels, C, penalty='l2', cv=5)
        results.append({
            'Model': 'Sentence Embeddings + Logistic Regression',
            'Regularization': 'L2',
            'C': C,
            'Mean CV Accuracy': round(mean_emb, 4),
            'Std Dev': round(std_emb, 4),
            'Non-zero Features': '-'
        })
    
    df = pd.DataFrame(results)
    return df


def get_best_models_summary():
    """
    Return a summary table of the best model for each configuration.
    """
    reviews, labels = get_reviews_and_labels()
    
    results = []
    
    scores_l1, mean_l1, std_l1 = evaluate_tfidf_cv(reviews, labels, 0.001, penalty='l1', cv=5)
    results.append({
        'Model': 'TF-IDF + L1',
        'Best C': 0.001,
        'Mean CV Accuracy': round(mean_l1, 4),
        'Std Dev': round(std_l1, 4),
        'Notes': 'Strong regularization; may have few features'
    })
    
    scores_l2, mean_l2, std_l2 = evaluate_tfidf_cv(reviews, labels, 100, penalty='l2', cv=5)
    results.append({
        'Model': 'TF-IDF + L2',
        'Best C': 100,
        'Mean CV Accuracy': round(mean_l2, 4),
        'Std Dev': round(std_l2, 4),
        'Notes': 'Weak regularization; dense weights'
    })
    
    scores_emb, mean_emb, std_emb = evaluate_embedding_cv(reviews, labels, 1, penalty='l2', cv=5)
    results.append({
        'Model': 'Embeddings + L2',
        'Best C': 1,
        'Mean CV Accuracy': round(mean_emb, 4),
        'Std Dev': round(std_emb, 4),
        'Notes': 'Best overall; semantic understanding'
    })
    
    df = pd.DataFrame(results)
    return df
