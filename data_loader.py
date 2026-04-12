def get_reviews_and_labels():
    reviews = [
        "Amazing movie with great acting",
        "Great plot and amazing story",
        "Loved it, fantastic experience",
        "Brilliant acting and excellent script",
        "Wonderful film, I enjoyed every minute",
        "This was a great movie",
        "The movie was amazing",
        "Super fun and entertaining",
        "It was okay, not bad",
        "Not terrible, actually fine",
        "Decent movie with good acting",
        "Pretty good overall",
        "Terrible acting and boring plot",
        "Waste of time",
        "I regret watching this movie",
        "Awful movie with terrible dialogue",
        "Boring story and bad acting",
        "Horrible plot, I hated it",
        "So bad I fell asleep",
        "Disappointing and frustrating",
        "Not good, waste of time",
        "Good acting but terrible plot",
        "Amazing acting but boring story",
        "Great cast but the movie was terrible",
    ]

    labels = [
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]

    return reviews, labels
