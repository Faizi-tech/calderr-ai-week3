import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base
sentences = [
    "Python is a programming language.",
    "Artificial Intelligence is changing the world.",
    "Dogs are loyal pets.",
    "The Eiffel Tower is in Paris.",
    "Football is the world's most popular sport."
]

# Generate embeddings for the knowledge base
embeddings = model.encode(sentences)

# Convert embeddings to float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

print("Embeddings Shape:", embeddings.shape)
print("Data Type:", embeddings.dtype)

# Get embedding dimension
dimension = embeddings.shape[1]
print(f"Embedding Dimension: {dimension}")

# Create a FAISS index
index = faiss.IndexFlatL2(dimension)

print("FAISS index created successfully!")
print(f"Vectors in index: {index.ntotal}")

# Add embeddings to the FAISS index
index.add(embeddings)

print("\nEmbeddings added successfully!")
print(f"Vectors in index: {index.ntotal}")

# Get a search query from the user
query = input("\nEnter your search query: ")

# Generate embedding for the query
query_embedding = model.encode([query])

# Convert query embedding to float32
query_embedding = np.array(query_embedding).astype("float32")

# Number of nearest neighbors
k = 3

# Search the FAISS index
distances, indices = index.search(query_embedding, k)

# Display the results
print("\nTop 3 Matching Sentences:\n")

for rank, sentence_index in enumerate(indices[0], start=1):
    print(f"{rank}. {sentences[sentence_index]}")
    print(f"   Distance: {distances[0][rank - 1]:.4f}\n")