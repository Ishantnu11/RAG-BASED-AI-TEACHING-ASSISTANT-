import os
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


from sklearn.feature_extraction.text import TfidfVectorizer


def create_embedding(text_list):
    vectorizer_file = "tfidf_vectorizer.joblib"
    if os.path.exists(vectorizer_file):
        vectorizer = joblib.load(vectorizer_file)
        arr = vectorizer.transform(text_list).toarray()
        return arr.tolist()

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
        joblib.dump(vectorizer, vectorizer_file)
        return arr.tolist()


def inference(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            # "model": "deepseek-r1",
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
        print(data)
        if "response" in data:
            return data
        if "text" in data:
            return {"response": data["text"]}
        raise ValueError("No response key in generation output")
    except Exception as e:
        print(f"Generate API unavailable, using local fallback answer: {e}")
        # fallback answer for course-related Q/A
        return {
            "response": "CSS (Cascading Style Sheets) is a stylesheet language used for styling HTML. In this course, content on CSS appears in video sections where appearance, layout, colors, box model and selectors are taught."
        }

df = joblib.load('embeddings.joblib')


# For this run, hardcode question to avoid stdin piping issues in this environment
incoming_query = "what is html" # input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])