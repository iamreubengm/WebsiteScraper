# # rag_src/retrieve.py
# import faiss, pickle, json, numpy as np
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi

# FAISS_FILE = "data/faiss.index"
# BM25_FILE = "data/bm25.pkl"
# CHUNK_FILE = "data/chunks.jsonl"

# print("Loading indexes and model…")
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# index = faiss.read_index(FAISS_FILE)
# with open(BM25_FILE, "rb") as f:
#     bm25_data = pickle.load(f)
# bm25, bm25_ids = bm25_data["bm25"], bm25_data["ids"]

# def hybrid_search(query, top_k=5, alpha=0.5):
#     q_emb = model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_emb)
#     sims, idxs = index.search(q_emb, top_k)
#     dense_scores = {bm25_ids[i]: sims[0][j] for j, i in enumerate(idxs[0])}

#     tokenized_query = query.split()
#     bm25_scores = bm25.get_scores(tokenized_query)
#     top_bm25 = np.argsort(bm25_scores)[::-1][:top_k]
#     sparse_scores = {bm25_ids[i]: bm25_scores[i] for i in top_bm25}

#     all_ids = set(dense_scores) | set(sparse_scores)
#     combined = {
#         i: alpha * dense_scores.get(i, 0) + (1 - alpha) * sparse_scores.get(i, 0)
#         for i in all_ids
#     }
#     ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
#     return ranked[:top_k]

# print(hybrid_search("Who is Nathan?"))


# rag_src/retrieve.py
import faiss, pickle, json, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

FAISS_FILE = "data/faiss.index"
BM25_FILE = "data/bm25.pkl"
CHUNK_FILE = "data/chunks.jsonl"

print("Loading indexes and model…")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(FAISS_FILE)

with open(BM25_FILE, "rb") as f:
    bm25_data = pickle.load(f)
bm25, bm25_ids = bm25_data["bm25"], bm25_data["ids"]

# Load all chunk texts into a dictionary
print("Loading chunk texts…")
id_to_text = {}
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        id_to_text[rec["doc_id"]] = rec["text"]

def hybrid_search(query, top_k=10, alpha=0.5):
    # Dense
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    sims, idxs = index.search(q_emb, top_k)
    dense_scores = {bm25_ids[i]: sims[0][j] for j, i in enumerate(idxs[0])}

    # Sparse
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25 = np.argsort(bm25_scores)[::-1][:top_k]
    sparse_scores = {bm25_ids[i]: bm25_scores[i] for i in top_bm25}

    # Combine
    all_ids = set(dense_scores) | set(sparse_scores)
    combined = {
        i: alpha * dense_scores.get(i, 0) + (1 - alpha) * sparse_scores.get(i, 0)
        for i in all_ids
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, id_to_text.get(doc_id, ""), score) for doc_id, score in ranked[:top_k]]

# Example query
query = "Who is Nathan Davis?"
results = hybrid_search(query, top_k=10)

print("\n🔍 Query:", query)
for i, (doc_id, text, score) in enumerate(results, 1):
    print(f"\n[{i}] {doc_id} (score={score:.4f})")
    print(text[:400], "..." if len(text) > 400 else "")

