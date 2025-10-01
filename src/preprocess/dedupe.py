# src/preprocess/dedupe.py
import json, hashlib, os
from collections import defaultdict
from tqdm import tqdm

IN_PATH = "data_clean/documents.jsonl"
OUT_PATH = "data_clean/documents.dedup.jsonl"
MAP_PATH = "data_clean/dedup_map.tsv"

def shingles(text, k=3):
    text = " ".join(text.split())
    return { text[i:i+k] for i in range(max(0, len(text)-k+1)) }

def jaccard(a, b):
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def doc_fingerprint(doc, k=3):
    text = "\n\n".join(s["text"] for s in doc.get("sections", []))
    return shingles(text, k=k)

def main(threshold=0.9):
    # Simple O(n^2) pass with bucketing by domain to keep it manageable initially.
    docs = []
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    keep = []
    dup_map = []
    seen_fp = []

    for d in tqdm(docs, desc="dedupe"):
        fp = doc_fingerprint(d)
        is_dup = False
        for i, (other_fp, other_id) in enumerate(seen_fp):
            if jaccard(fp, other_fp) >= threshold:
                dup_map.append((d["doc_id"], other_id))
                is_dup = True
                break
        if not is_dup:
            keep.append(d)
            seen_fp.append((fp, d["doc_id"]))

    # write outputs
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for d in keep:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(MAP_PATH, "w", encoding="utf-8") as m:
        m.write("dup_id\tkept_id\n")
        for a,b in dup_map:
            m.write(f"{a}\t{b}\n")

    print(f"[dedupe] kept {len(keep)} / {len(docs)} (threshold={threshold})")
    print(f"[dedupe] map → {MAP_PATH}")

if __name__ == "__main__":
    main()
