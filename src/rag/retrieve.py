import faiss, pickle, json, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import spacy

FAISS_FILE = "data/faiss.index"
BM25_FILE = "data/bm25.pkl"
CHUNK_FILE = "data/chunks.jsonl"

print("Loading embedding model and indexes...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
nlp = spacy.load("en_core_web_sm")

index = faiss.read_index(FAISS_FILE)
with open(BM25_FILE, "rb") as f:
    bm25_data = pickle.load(f)
bm25, bm25_ids = bm25_data["bm25"], bm25_data["ids"]

id_to_text = {}
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading chunks"):
        rec = json.loads(line)
        id_to_text[rec["doc_id"]] = rec["text"]

def normalize_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, ent.text.lower())
    return text

def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
    combined = {}
    for rank_list in [dense_ranks, sparse_ranks]:
        for rank, (doc_id, _) in enumerate(rank_list):
            combined[doc_id] = combined.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query, top_k=10):
    query = normalize_entities(query)
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    sims, idxs = index.search(q_emb, 50)
    dense_ranked = [(bm25_ids[i], sims[0][j]) for j, i in enumerate(idxs[0])]
    tokenized = query.split()
    bm25_scores = bm25.get_scores(tokenized)
    top_bm25 = np.argsort(bm25_scores)[::-1][:50]
    sparse_ranked = [(bm25_ids[i], bm25_scores[i]) for i in top_bm25]
    fused = reciprocal_rank_fusion(dense_ranked, sparse_ranked)
    top_docs = fused[:top_k]
    return [(doc_id, id_to_text[doc_id], score) for doc_id, score in top_docs]
