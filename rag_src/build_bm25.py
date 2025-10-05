# rag_src/build_bm25.py
import json, pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

CHUNK_FILE = "data/chunks.jsonl"
BM25_FILE = "data/bm25.pkl"

docs, ids = [], []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading chunks"):
        rec = json.loads(line)
        ids.append(rec["doc_id"])
        docs.append(rec["text"].split())

bm25 = BM25Okapi(docs)

with open(BM25_FILE, "wb") as f:
    pickle.dump({"bm25": bm25, "ids": ids}, f)

print(f"✅ BM25 index built for {len(ids)} chunks.")
