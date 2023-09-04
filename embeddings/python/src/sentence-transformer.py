import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
  "That is a very happy person",
  "That is a happy dog",
  "Today is a sunny day"
]
embeddings = model.encode(sentences)

query = "That is a happy person"
query_embedding = model.encode(query)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print(f'Query: {query}')
for e, s in zip(embeddings, sentences):
    print(s, " -> similarity score = ",
         cosine_similarity(e, query_embedding))
