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