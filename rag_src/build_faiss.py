# src/build_faiss.py
import json, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/chunks.jsonl"
INDEX_FILE = "data/faiss.index"
EMBED_FILE = "data/chunk_embeddings.npy"

print("Loading model…")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts, ids = [], []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading chunks"):
        rec = json.loads(line)
        texts.append(rec["text"])
        ids.append(rec["doc_id"])

print("Encoding texts…")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

print("Building FAISS index…")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
np.save(EMBED_FILE, embeddings)
print(f"✅ FAISS index built for {len(ids)} chunks.")
