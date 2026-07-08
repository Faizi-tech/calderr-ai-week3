from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("BAAI/bge-small-en")

# Knowledge base
sentences = [
    "Python is a popular programming language.",
    "Java is widely used for enterprise applications.",
    "C++ is known for high performance.",
    "JavaScript powers interactive websites.",
    "Artificial Intelligence enables machines to learn.",
    "Machine learning is a subset of Artificial Intelligence.",
    "Deep learning uses neural networks.",
    "Data science combines statistics and programming.",
    "Cloud computing provides scalable computing resources.",
    "Cybersecurity protects computer systems from attacks.",

    "Cats are popular household pets.",
    "Dogs are loyal and friendly animals.",
    "Elephants are the largest land mammals.",
    "Lions are known as the kings of the jungle.",
    "Tigers are powerful predators.",
    "Dolphins are intelligent marine mammals.",
    "Whales are the largest animals on Earth.",
    "Penguins cannot fly but are excellent swimmers.",
    "Eagles have excellent eyesight.",
    "Butterflies undergo complete metamorphosis.",

    "The Earth revolves around the Sun.",
    "The Moon is Earth's natural satellite.",
    "Mars is called the Red Planet.",
    "Jupiter is the largest planet in the Solar System.",
    "Saturn is famous for its rings.",
    "Venus is the hottest planet.",
    "Mercury is the closest planet to the Sun.",
    "Neptune is an ice giant.",
    "The Milky Way is our galaxy.",
    "Astronauts explore space using spacecraft.",

    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from many locations on Earth.",
    "Mount Everest is the highest mountain above sea level.",
    "The Amazon Rainforest is the largest tropical rainforest.",
    "The Nile is one of the world's longest rivers.",
    "Tokyo is the capital of Japan.",
    "Islamabad is the capital of Pakistan.",
    "New York City is famous for Times Square.",
    "London is home to Big Ben.",
    "Rome is known for the Colosseum.",

    "Football is the world's most popular sport.",
    "Cricket is widely played in South Asia.",
    "Basketball was invented by James Naismith.",
    "Tennis is played with rackets.",
    "Swimming is a full-body exercise.",
    "The Olympic Games are held every four years.",
    "Lionel Messi is a famous football player.",
    "Virat Kohli is a renowned cricketer.",
    "Chess is a game of strategy.",
    "Table tennis is also called ping pong.",

    "Water boils at 100 degrees Celsius at sea level.",
    "Ice melts at 0 degrees Celsius.",
    "Plants produce oxygen through photosynthesis.",
    "The human heart pumps blood throughout the body.",
    "The brain controls the nervous system.",
    "DNA carries genetic information.",
    "Vaccines help prevent infectious diseases.",
    "The human skeleton consists of 206 bones.",
    "Exercise improves physical health.",
    "Healthy diets include fruits and vegetables.",

    "The Internet connects billions of devices.",
    "Wi-Fi enables wireless networking.",
    "5G provides faster mobile communication.",
    "Databases store structured information.",
    "SQL is used to query relational databases.",
    "Git is a version control system.",
    "GitHub hosts software repositories.",
    "Linux is an open-source operating system.",
    "Windows is developed by Microsoft.",
    "OpenAI develops advanced AI models.",

    "The Taj Mahal is located in India.",
    "The Pyramids of Giza are ancient wonders.",
    "The Pacific Ocean is the largest ocean.",
    "The Atlantic Ocean separates Europe and America.",
    "The Sahara is the largest hot desert.",
    "Volcanoes erupt molten lava.",
    "Earthquakes occur due to tectonic plate movement.",
    "Rainbows are formed by light refraction.",
    "Lightning is an electrical discharge.",
    "Hurricanes are powerful tropical storms.",

    "Reading books improves knowledge.",
    "Education develops critical thinking.",
    "Music can improve mood.",
    "Painting is a form of visual art.",
    "Photography captures moments.",
    "Travel exposes people to different cultures.",
    "Libraries provide access to knowledge.",
    "Scientists conduct experiments to test hypotheses.",
    "Engineers solve practical problems.",
    "Mathematics is the language of science.",

    "Renewable energy includes solar and wind power.",
    "Solar panels convert sunlight into electricity.",
    "Wind turbines generate clean energy.",
    "Electric vehicles reduce fuel consumption.",
    "Recycling helps protect the environment.",
    "Plastic pollution harms marine life.",
    "Forests absorb carbon dioxide.",
    "Climate change affects global weather patterns.",
    "Conservation protects endangered species.",
    "Sustainable development balances growth and environmental protection."
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