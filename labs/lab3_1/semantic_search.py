from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base
sentences = [
    "Python is a programming language.",
    "Artificial Intelligence is transforming industries.",
    "Machine learning is a branch of AI.",
    "Football is the world's most popular sport.",
    "Cats are wonderful pets.",
    "Deep learning uses neural networks.",
    "The Eiffel Tower is located in Paris.",
    "Cloud computing provides scalable resources.",
    "Data science combines statistics and programming.",
    "Cybersecurity protects computer systems from attacks."
]

print("Knowledge base loaded successfully!")
print(f"Total Sentences: {len(sentences)}")

# Generate embeddings for the knowledge base
embeddings = model.encode(sentences)

print("\nEmbeddings generated successfully!")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape}")

# Take query from the user
query = input("\nEnter your search query: ")

# Generate embedding for the query
query_embedding = model.encode(query)

print("\nQuery embedding generated successfully!")
print(f"Embedding Shape: {query_embedding.shape}")

# Calculate cosine similarity
similarity_scores = cosine_similarity(
    [query_embedding], embeddings
)[0]

# Find the index of the most similar sentence
best_match_index = similarity_scores.argmax()

# Display the result
print("\nBest Match:")
print(sentences[best_match_index])

print(f"Similarity Score: {similarity_scores[best_match_index]:.4f}")