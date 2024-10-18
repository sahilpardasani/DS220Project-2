import chromadb
from sentence_transformers import SentenceTransformer

# Load a compatible sentence transformer model for embeddings
model_name = 'all-MiniLM-L6-v2'  # This is a small, efficient model compatible with most systems
embedding_model = SentenceTransformer(model_name)

# Create a class-based embedding function following the new signature
class EmbeddingFunction:
    def __call__(self, input):
        return embedding_model.encode(input).tolist()

# Initialize the embedding function
embedding_function = EmbeddingFunction()

# Create a ChromaDB client
client = chromadb.Client()

# Create a collection, passing the updated embedding function
collection = client.create_collection(
    name="my_collection", 
    embedding_function=embedding_function
)

# Add some sample documents to the collection
collection.add(
    documents=[
        "This document is about New York",
        "This document is about Delhi"
    ],
    ids=['id1', 'id2']
)

# Retrieve all documents from the collection
all_docs = collection.get()

# Print the results to verify the correct functioning
print(all_docs)

documents=collection.get(ids=["id1"])
print(documents)

#results = collection.query(
#    query_texts=["Query is about Chhole Bhature"],
 #   n_results=2
#)
#print(results)

collection.delete(ids=all_docs['ids'])
collection.get()

collection.add(
    documents=[
        "This document is about New York",
        "This document is about Delhi"
    ],
    ids=['id3', 'id4'],
    metadatas=[
        {"url": "https://en.wikipedia.org/wiki/New_York_City"},
        {"url": "https://en.wikipedia.org/wiki/New_Delhi"}
    ]
)

results = collection.query(
    query_texts=["Query is about Chhole Bhature"],
    n_results=2
)
print(results)