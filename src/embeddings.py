from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# List of sentences
sentences = [
    "I love programming.",
    "Coding is my favorite hobby.",
    "I enjoy playing football.",
    "Artificial Intelligence is transforming the world."
]

# Generate embeddings
embeddings = model.encode(sentences)

# Display each sentence and its embedding details
for i, (sentence, embedding) in enumerate(zip(sentences, embeddings), start=1):
    print(f"\nSentence {i}: {sentence}")
    print(f"Embedding Shape: {embedding.shape}")
    print(f"First 10 Values: {embedding[:10]}")

print("\nCosine Similarity Matrix:\n")

similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix)