from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#C_values list for experiments
C_values = [0.01, 0.1, 1, 10, 100]

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

X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42, stratify=labels)
print("Training data:", len(X_train))
print("Testing data:", len(X_test))

#c=see if stratify worked correctly
print("Train positives:", sum(y_train), "out of", len(y_train))
print("Test positives:", sum(y_test), "out of", len(y_test))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
    )
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Train shape:", X_train_vec.shape)
print("Test shape:", X_test_vec.shape)
print("Number of features:", len(vectorizer.get_feature_names_out()))

print("\n" + "="*60)
print("Logistic Regression with L1 Regularization")

for C in C_values:
    print("\n" + "="*60)
    print(f"C value: {C}")

    model = LogisticRegression(
        max_iter=1000, 
        C=C,
        solver = "liblinear",
        penalty='l1'
        )
    model.fit(X_train_vec, y_train)

    train_pred = model.predict(X_train_vec)
    test_pred = model.predict(X_test_vec)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    print("Train Accuracy:", f"{train_accuracy:.3f}")
    print("Test Accuracy:", f"{test_accuracy:.3f}")

    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    word_weights = list(zip(feature_names, weights))
    word_weights_sorted = sorted(word_weights, key=lambda x: x[1], reverse=True)
    print("\nTop 10 Positive Words:")
    for word, weight in word_weights_sorted[:10]:
        print(f"Word: {word}, Weight: {weight:+.4f}")
    print("\nTop 10 Negative Words:")
    for word, weight in word_weights_sorted[-10:]:
        print(f"Word: {word}, Weight: {weight:+.4f}")
    

print("\n" + "="*60)
print("Logistic Regression with L2 Regularization")

for C in C_values:
    print("\n" + "="*60)
    print(f"C value: {C}")

    model2 = LogisticRegression(
        max_iter=1000, 
        C=C,
        )
    model2.fit(X_train_vec, y_train)

    train_pred = model2.predict(X_train_vec)
    test_pred = model2.predict(X_test_vec)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    print("Train Accuracy:", f"{train_accuracy:.3f}")
    print("Test Accuracy:", f"{test_accuracy:.3f}")

    feature_names = vectorizer.get_feature_names_out()
    weights = model2.coef_[0]
    word_weights = list(zip(feature_names, weights))
    word_weights_sorted = sorted(word_weights, key=lambda x: x[1], reverse=True)
    print("\nTop 10 Positive Words:")
    for word, weight in word_weights_sorted[:10]:
        print(f"Word: {word}, Weight: {weight:+.4f}")
    print("\nTop 10 Negative Words:")
    for word, weight in word_weights_sorted[-10:]:
        print(f"Word: {word}, Weight: {weight:+.4f}")
    


