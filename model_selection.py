from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings

# Hide sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

C_values = [-0.001, -0.01, -0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

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

    # Mild/neutral positives
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

    # Negation + tricky mixed cases
    "Not good, waste of time",
    "Good acting but terrible plot",
    "Amazing acting but boring story",
    "Great cast but the movie was terrible",
]

labels: list[int] = [
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
]

# Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(reviews)
feature_names = vectorizer.get_feature_names_out()

score_list_l1 = []
score_list_l2 = []


def print_top_features(model, feature_names, top_n=10):
    """Print top positive and negative features from a fitted model."""
    coefs = model.coef_[0]

    top_positive_indices = coefs.argsort()[-top_n:][::-1]
    top_negative_indices = coefs.argsort()[:top_n]

    print("\nTop 10 Positive Words:")
    for idx in top_positive_indices:
        print(f"{feature_names[idx]}: {coefs[idx]:+.4f}")

    print("\nTop 10 Negative Words:")
    for idx in top_negative_indices:
        print(f"{feature_names[idx]}: {coefs[idx]:+.4f}")

    nonzero_features = (coefs != 0).sum()
    print(f"\nNon-zero features: {nonzero_features} / {len(feature_names)}")


print("L2 Regularization Cross-Validation Scores:")
print("=" * 60)

for C in C_values:
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        C=C
    )

    scores = cross_val_score(model, X_vec, labels, cv=5)
    score_list_l2.append(scores.mean())

    print(f"C={C}: Fold accuracies: {scores}, Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print("-" * 60)

    # Fit on full dataset so we can inspect learned coefficients
    model.fit(X_vec, labels)
    print_top_features(model, feature_names)
    print("=" * 60)

highest_c_l2 = C_values[score_list_l2.index(max(score_list_l2))]
print(f"Best C for L2: {highest_c_l2} with mean accuracy {max(score_list_l2):.4f}")
print("\n" + "=" * 60)

print("L1 Regularization Cross-Validation Scores:")
print("=" * 60)

for C in C_values:
    model2 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        C=C
    )

    scores2 = cross_val_score(model2, X_vec, labels, cv=5)
    score_list_l1.append(scores2.mean())

    print(f"C={C}: Fold accuracies: {scores2}, Mean: {scores2.mean():.4f}, Std: {scores2.std():.4f}")
    print("-" * 60)

    # Fit on full dataset so we can inspect learned coefficients
    model2.fit(X_vec, labels)
    print_top_features(model2, feature_names)
    print("=" * 60)

highest_c_l1 = C_values[score_list_l1.index(max(score_list_l1))]
print(f"Best C for L1: {highest_c_l1} with mean accuracy {max(score_list_l1):.4f}")
print("\n" + "=" * 60)

if max(score_list_l1) > max(score_list_l2):
    print(f"L1 regularization performed better with C={highest_c_l1} achieving mean accuracy {max(score_list_l1):.4f}")
elif max(score_list_l2) > max(score_list_l1):
    print(f"L2 regularization performed better with C={highest_c_l2} achieving mean accuracy {max(score_list_l2):.4f}")
else:
    print(f"Both L1 and L2 regularization performed equally well with mean accuracy {max(score_list_l1):.4f}")