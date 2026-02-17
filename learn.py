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

reviewData: list[dict[str, int]] = []

weights: dict[str, int] = {
    "amazing": 5,
    "great": 4,
    "okay": 2,
    "fine": 2,
    "nothing": 1,
    "waste": -5,
    "terrible": -5, 
    "boring": -4,
    "regret": -5

}

# Labels
# 1 = positive, 0 = negative
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

for review in reviews:
    review = review.lower()
    words = review.split()
    wordCount: dict[str, int] = {}
    for word in words:
        if word in wordCount:
            wordCount[word] += 1
        else:
            wordCount[word] = 1
    reviewData.append(wordCount)


def scoreReview(reviewData: list[dict[str, int]], weights: dict[str, int]) -> int:
    score = 0
    for review in reviewData:
        for word, count in review.items():
            if word in weights:
                score += weights[word] * count
    return score
    
def predict(score: int) -> int:
    if score >= 0:
        return 1  # positive
    else:
        return 0  # negative

def finalTest(reviewData: list[dict[str, int]], weights: dict[str, int], labels: list[int]) -> None:
    for i in range(len(reviewData)):
        score = scoreReview([reviewData[i]], weights)
        prediction = predict(score)
        print(f"Review: {reviews[i]}")
        print(f"Score: {score}, Prediction: {'Positive' if prediction == 1 else 'Negative'}, Actual: {'Positive' if labels[i] == 1 else 'Negative'}")
        print()

finalTest(reviewData, weights, labels)