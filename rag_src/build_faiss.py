# # src/build_faiss.py
# import json, faiss, numpy as np
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer

# CHUNK_FILE = "data/chunks.jsonl"
# INDEX_FILE = "data/faiss.index"
# EMBED_FILE = "data/chunk_embeddings.npy"

# print("Loading model…")
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# texts, ids = [], []
# with open(CHUNK_FILE, "r", encoding="utf-8") as f:
#     for line in tqdm(f, desc="Reading chunks"):
#         rec = json.loads(line)
#         texts.append(rec["text"])
#         ids.append(rec["doc_id"])

# print("Encoding texts…")
# embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# # normalize for cosine similarity
# faiss.normalize_L2(embeddings)

# print("Building FAISS index…")
# index = faiss.IndexFlatIP(embeddings.shape[1])
# index.add(embeddings)

# faiss.write_index(index, INDEX_FILE)
# np.save(EMBED_FILE, embeddings)
# print(f"✅ FAISS index built for {len(ids)} chunks.")


# rag_src/build_faiss.py
import os, json, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/chunks.jsonl"
INDEX_FILE = "data/faiss.index"
EMBED_FILE = "data/chunk_embeddings.npy"

# ✅ Create output directory if it doesn’t exist
os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

# ✅ Use a stronger embedding model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

texts, ids = [], []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading chunks"):
        rec = json.loads(line)
        texts.append(rec["text"])
        ids.append(rec["doc_id"])

# ✅ Encode with normalization
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(EMBED_FILE, embeddings)

# ✅ Build FAISS index (Inner Product)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

print(f"✅ FAISS index built for {len(embeddings)} chunks using BGE-large-en-v1.5.")
