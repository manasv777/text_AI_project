from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reviews dataset
reviews: list[str] = [
    "The movie was amazing",
    "Amazing movie with great acting",
    "I had an amazing time watching this",
    "This was a great movie",
    "The movie was okay",
    "It was fine nothing special",
    "Waste of time",
    "The movie was terrible",
    "Terrible acting and boring plot",
    "I regret watching this movie"
]

labels: list[int] = [
    1,  # amazing
    1,  # amazing + great
    1,  # amazing
    1,  # great
    1,  # weak positive
    1,  # weak positive
    0,  # waste
    0,  # terrible
    0,  # terrible + boring
    0   # regret
]

X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
print("Training data:", len(X_train))
print("Testing data:", len(X_test))

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Train shape:", X_train_vec.shape)
print("Test shape:", X_test_vec.shape)
print("Number of features:", len(vectorizer.get_feature_names_out()))

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Printing the process out so you can see it live
print("\n--- Test Predictions ---")
for review, pred, actual in zip(X_test, y_pred, y_test):
    pred_text = "Positive" if pred == 1 else "Negative"
    actual_text = "Positive" if actual == 1 else "Negative"
    print(f"Review: {review}")
    print(f"Prediction: {pred_text}, Actual: {actual_text}\n")


