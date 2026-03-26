import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "embeddings" in data:
            return data["embeddings"]
        raise ValueError("No embeddings key in API response")
    except Exception as e:
        print(f"Embedding API unavailable, using local TF-IDF fallback: {e}")
        vectorizer = TfidfVectorizer()
        arr = vectorizer.fit_transform(text_list).toarray()
        joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
        return arr.tolist()


jsons = os.listdir("jsons")  # List all the jsons 
all_chunks = []
all_texts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Loading chunks for {json_file}")
    for chunk in content['chunks']:
        chunk['chunk_id'] = chunk_id
        all_chunks.append(chunk)
        all_texts.append(chunk['text'])
        chunk_id += 1

# Create embeddings for all chunks together to keep vector dimension stable
print("Creating embeddings for all chunks")
all_embeddings = create_embedding(all_texts)
if len(all_embeddings) != len(all_chunks):
    raise ValueError("Embedding count mismatch with chunk count")

for chunk, embedding in zip(all_chunks, all_embeddings):
    chunk['embedding'] = embedding

# Build dataframe
my_dicts = all_chunks

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')

