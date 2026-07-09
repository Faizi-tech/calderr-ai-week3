import chromadb

# Create a ChromaDB client
client = chromadb.Client()

print("ChromaDB client created successfully!")

# Create a collection
collection = client.create_collection(name="knowledge_base")

print("Collection created successfully!")
print(f"Collection Name: {collection.name}")