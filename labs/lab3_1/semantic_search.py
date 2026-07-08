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

while True:
    query = input("\nEnter your search query (or type 'exit' to quit): ")

    if query.lower() == "exit":
        print("\nThank you for using Semantic Search CLI!")
        break

    # Generate embedding for the query
    query_embedding = model.encode(query)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        [query_embedding], embeddings
    )[0]

    # Get top 3 matches
    top_k = 3
    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    print("\nTop 3 Matching Sentences:\n")

    for rank, index in enumerate(top_indices, start=1):
        print(f"{rank}. {sentences[index]}")
        print(f"   Similarity Score: {similarity_scores[index]:.4f}\n")