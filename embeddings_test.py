from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Amazing movie with great acting",
    "Terrible acting and boring plot"
]

embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
print("Embedding for first sentence:", embeddings[0])
print("Embedding for second sentence:", embeddings[1])  