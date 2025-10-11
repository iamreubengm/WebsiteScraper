import os, json, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/chunks.jsonl"
INDEX_FILE = "data/faiss.index"
EMBED_FILE = "data/chunk_embeddings.npy"

os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

texts, ids = [], []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading chunks"):
        rec = json.loads(line)
        texts.append(rec["text"])
        ids.append(rec["doc_id"])

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(EMBED_FILE, embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

print(f"FAISS index built for {len(embeddings)} chunks using BGE-large-en-v1.5.")
