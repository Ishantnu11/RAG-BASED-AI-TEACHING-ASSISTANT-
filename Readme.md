# How to use this RAG AI Teaching assistant on your own data
RAG AI Teaching Assistant
This project is a Retrieval-Augmented Generation (RAG) system designed to act as a teaching assistant for video-based courses. It processes video content, converts it into searchable vector embeddings, and uses an LLM to answer student queries based on the specific context of the lessons.

🚀 How it Works
The assistant follows a 5-step pipeline to transform raw video data into an intelligent conversational interface.

Step 1: Collect Your Videos
Gather all your course video files (e.g., .mp4, .mkv) and place them into the videos/ directory.

Step 2: Extract Audio (video_to_mp3.py)
Run the script to convert video files into MP3 format using ffmpeg. This makes the transcription process more efficient.

Command: python video_to_mp3.py

Step 3: Transcribe to JSON (mp3_to_json.py)
Convert the audio files into text transcripts. The output is stored as JSON files containing timestamps and text chunks, which are essential for referencing specific parts of a video.

Command: python mp3_to_json.py

Step 4: Generate Vector Embeddings (preprocess_json.py)
This step converts the text chunks into mathematical vectors (embeddings) so the AI can "search" by meaning rather than just keywords.

It uses the bge-m3 model via a local Ollama instance.

Fallback: If the API is offline, it automatically falls back to a local TF-IDF vectorizer.

The final data is saved as a joblib pickle file for fast loading.

Step 5: Query & Response (process_incoming.py)
When a user asks a question (e.g., "What is HTML?"), the system:

Loads the joblib vector database.

Finds the most relevant video chunks using cosine similarity.

Constructs a prompt and feeds it to the LLM.

Saves the final answer to response.txt.

🛠 Prerequisites
FFmpeg: Required for audio conversion.

Ollama: Required for running the bge-m3 embedding model locally.

Python Libraries: ```bash
pip install pandas numpy scikit-learn joblib requests
