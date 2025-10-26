import os
import pdfplumber
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "astrogigs")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

def load_books(data_folder="data"):
    texts = []
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif file.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + " "
                texts.append(full_text)
    return texts

def split_documents(texts, chunk_size=500, chunk_overlap=50):
    docs = []
    for text in texts:
        start = 0
        while start < len(text):
            end = start + chunk_size
            docs.append(text[start:end])
            start += chunk_size - chunk_overlap
    return docs


def create_embeddings(docs):
    embeddings = model.encode(docs, show_progress_bar=True)
    return embeddings

def upload_to_pinecone(docs, embeddings, batch_size=50):
    for i in range(0, len(docs), batch_size):
        batch_vectors = []
        for j, emb in enumerate(embeddings[i:i + batch_size]):
            batch_vectors.append({
                "id": f"chunk-{i+j}",
                "values": emb.tolist(),
                "metadata": {"text": docs[i+j]}
            })
        index.upsert(vectors=batch_vectors)
    print(f" Uploaded {len(docs)} chunks to Pinecone index '{INDEX_NAME}'")

def query_rag(query, top_k=3):
    query_emb = model.encode([query])[0].tolist()
    res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    context = " ".join([m["metadata"]["text"] for m in matches if "metadata" in m])
    return context

if __name__ == "__main__":
    print("Loading documents")
    texts = load_books("data")
    print("Splitting documents into chunks")
    docs = split_documents(texts)
    print(f"Total chunks: {len(docs)}")

    print("Creating embeddings")
    embeddings = create_embeddings(docs)

    print("Uploading chunks to Pinecone")
    upload_to_pinecone(docs, embeddings)






# import pinecone
# import os
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = os.getenv("INDEX_NAME", "astrogigs")
# PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# # initialize index directly with API key
# index = pinecone.Index(index_name=INDEX_NAME, environment=PINECONE_ENV)

# # load embedding model
# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# # query function
# def query_rag(query, top_k=3):
#     query_emb = model.encode([query])[0].tolist()
#     res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
#     matches = res.get("matches", [])
#     context = " ".join([m["metadata"]["text"] for m in matches if "metadata" in m])
#     return context
