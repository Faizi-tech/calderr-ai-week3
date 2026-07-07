from sentence_transformers import SentenceTransformer

# Load the pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentence to convert into an embedding
sentence = "Artificial Intelligence is transforming the world."

# Generate the embedding
embedding = model.encode(sentence)

# Display the embedding vector
print("Embedding Vector:")
print(embedding)

# Display the size (dimensions) of the embedding
print(f"\nEmbedding Shape: {embedding.shape}")