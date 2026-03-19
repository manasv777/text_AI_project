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

embedder = SentenceTransformer('all-MiniLM-L6-v2')

X = embedder.encode(reviews)

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),  # Unigrams and bigrams
    stop_words='english',  # Remove common English stop words
    sublinear_tf=True,  # Use sublinear term frequency scaling
)

X_vec = vectorizer.fit_transform(reviews)
model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42, C=3)
scores = cross_val_score(model, X, labels, cv=5)

print("Embeddings  Cross-Validation Scores:")
print("=" * 60)
print("Fold accuracies:", scores)
print("Mean accuracy:", scores.mean())
print("Std dev:", scores.std())

model.fit(X, labels)
print("Non-zero features:", np.sum(model.coef_ != 0))