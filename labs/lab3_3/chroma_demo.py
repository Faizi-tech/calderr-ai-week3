import chromadb

# Create a ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

print("ChromaDB client created successfully!")

# Create a collection
collection = client.get_or_create_collection(name="knowledge_base")

print("Collection created successfully!")
print(f"Collection Name: {collection.name}")

# Documents to store
documents = [
    "Python is a programming language.",
    "Artificial Intelligence is changing the world.",
    "Dogs are loyal pets.",
    "The Eiffel Tower is in Paris.",
    "Football is the world's most popular sport."
]

# Unique IDs for each document
ids = [
    "doc1",
    "doc2",
    "doc3",
    "doc4",
    "doc5"
]

# Add documents to the collection
if collection.count() == 0:
    collection.add(
        documents=documents,
        ids=ids
    )
    print("\nDocuments added successfully!")
else:
    print("\nCollection already contains documents.")

print(f"Total Documents: {collection.count()}")

# Search query
query = input("\nEnter your search query: ")

# Search the collection
results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\nTop 3 Matching Documents:\n")

for i in range(len(results["documents"][0])):
    print(f"{i + 1}. {results['documents'][0][i]}")
    print(f"   ID: {results['ids'][0][i]}")
    print(f"   Distance: {results['distances'][0][i]:.4f}\n")