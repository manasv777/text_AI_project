from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Reviews dataset
reviews: list[str] = [
    # Strong positives
    "Amazing movie with great acting",
    "Great plot and amazing story",
    "Loved it, fantastic experience",
    "Brilliant acting and excellent script",
    "Wonderful film, I enjoyed every minute",
    "This was a great movie",
    "The movie was amazing",
    "Super fun and entertaining",

    # Mild/neutral positives (still label as positive for now)
    "It was okay, not bad",
    "Not terrible, actually fine",
    "Decent movie with good acting",
    "Pretty good overall",

    # Strong negatives
    "Terrible acting and boring plot",
    "Waste of time",
    "I regret watching this movie",
    "Awful movie with terrible dialogue",
    "Boring story and bad acting",
    "Horrible plot, I hated it",
    "So bad I fell asleep",
    "Disappointing and frustrating",

    # Negation + tricky mixed cases (still keep labels simple)
    "Not good, waste of time",
    "Good acting but terrible plot",
    "Amazing acting but boring story",
    "Great cast but the movie was terrible",
]

labels: list[int] = [
    1,  # Amazing movie with great acting
    1,  # Great plot and amazing story
    1,  # Loved it, fantastic experience
    1,  # Brilliant acting and excellent script
    1,  # Wonderful film, I enjoyed every minute
    1,  # This was a great movie
    1,  # The movie was amazing
    1,  # Super fun and entertaining

    1,  # It was okay, not bad
    1,  # Not terrible, actually fine
    1,  # Decent movie with good acting
    1,  # Pretty good overall

    0,  # Terrible acting and boring plot
    0,  # Waste of time
    0,  # I regret watching this movie
    0,  # Awful movie with terrible dialogue
    0,  # Boring story and bad acting
    0,  # Horrible plot, I hated it
    0,  # So bad I fell asleep
    0,  # Disappointing and frustrating

    0,  # Not good, waste of time
    0,  # Good acting but terrible plot
    0,  # Amazing acting but boring story
    0,  # Great cast but the movie was terrible
]

C_values = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

X = embedder.encode(reviews)

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),  # Unigrams and bigrams
    stop_words='english',  # Remove common English stop words
    sublinear_tf=True,  # Use sublinear term frequency scaling
)

# Store best results for each model
results = {}

# Embeddings with L2
best_c_emb = None
best_mean_emb = -1
best_std_emb = None

for C in C_values:
    model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42, C=C)
    scores = cross_val_score(model, X, labels, cv=5)
    print(f"Embeddings  Cross-Validation Scores (C={C}):")
    print("=" * 60)
    print("Fold accuracies:", scores)
    print("Mean accuracy:", scores.mean())
    print("Std dev:", scores.std())
    print("Top features: Not interpretable for embeddings")
    print()
    if scores.mean() > best_mean_emb:
        best_mean_emb = scores.mean()
        best_std_emb = scores.std()
        best_c_emb = C

results['embeddings_l2'] = {'C': best_c_emb, 'mean': best_mean_emb, 'std': best_std_emb}

X_vec = vectorizer.fit_transform(reviews)

# TF-IDF with L1
best_c_l1 = None
best_mean_l1 = -1
best_std_l1 = None

for C in C_values:
    model2 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42, C=C)
    scores2 = cross_val_score(model2, X_vec, labels, cv=5)
    print(f"L1 Regularization Cross-Validation Scores (C={C}):")
    print("=" * 60)
    print("Fold accuracies:", scores2)
    print("Mean accuracy:", scores2.mean())
    print("Std dev:", scores2.std())
    
    # Fit model on full data to get coefficients
    model2.fit(X_vec, labels)
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model2.coef_[0]
    
    # Get top positive and negative features (excluding zeros for L1)
    non_zero_indices = coefficients != 0
    if np.sum(non_zero_indices) > 0:
        non_zero_coeffs = coefficients[non_zero_indices]
        non_zero_names = feature_names[non_zero_indices]
        
        top_positive_indices = non_zero_coeffs.argsort()[-5:][::-1]
        top_negative_indices = non_zero_coeffs.argsort()[:5]
        
        print("Top 5 positive features:")
        for idx in top_positive_indices:
            print(f"  {non_zero_names[idx]}: {non_zero_coeffs[idx]:.4f}")
        print("Top 5 negative features:")
        for idx in top_negative_indices:
            print(f"  {non_zero_names[idx]}: {non_zero_coeffs[idx]:.4f}")
    else:
        print("No non-zero features")
    print()
    
    if scores2.mean() > best_mean_l1:
        best_mean_l1 = scores2.mean()
        best_std_l1 = scores2.std()
        best_c_l1 = C

results['tfidf_l1'] = {'C': best_c_l1, 'mean': best_mean_l1, 'std': best_std_l1}

# TF-IDF with L2
best_c_l2 = None
best_mean_l2 = -1
best_std_l2 = None

for C in C_values:
    model3 = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42, C=C)
    scores3 = cross_val_score(model3, X_vec, labels, cv=5)
    print(f"L2 Regularization Cross-Validation Scores (C={C}):")
    print("=" * 60)
    print("Fold accuracies:", scores3)
    print("Mean accuracy:", scores3.mean())
    print("Std dev:", scores3.std())
    
    # Fit model on full data to get coefficients
    model3.fit(X_vec, labels)
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model3.coef_[0]
    
    # Get top positive and negative features
    top_positive_indices = coefficients.argsort()[-5:][::-1]
    top_negative_indices = coefficients.argsort()[:5]
    
    print("Top 5 positive features:")
    for idx in top_positive_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    print("Top 5 negative features:")
    for idx in top_negative_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    print()
    
    if scores3.mean() > best_mean_l2:
        best_mean_l2 = scores3.mean()
        best_std_l2 = scores3.std()
        best_c_l2 = C

results['tfidf_l2'] = {'C': best_c_l2, 'mean': best_mean_l2, 'std': best_std_l2}

# Print the best results
print("\nBest Results Summary:")
print("=" * 60)
for model, res in results.items():
    print(f"{model}: C={res['C']}, Mean Accuracy={res['mean']:.4f}, Std Dev={res['std']:.4f}")