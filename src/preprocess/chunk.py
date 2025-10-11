import os
import json
from nltk import sent_tokenize
from tqdm import tqdm

INPUT_DIR = "data_clean/text_corpus/by_doc"
OUTPUT_FILE = "data/chunks.jsonl"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def chunk_text(sentences, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, current = [], []
    num_words = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if num_words + sent_len > size:
            chunks.append(" ".join(current))
            # overlap
            overlap_words = " ".join(" ".join(current).split()[-overlap:])
            current = [overlap_words, sent]
            num_words = len(overlap_words.split()) + sent_len
        else:
            current.append(sent)
            num_words += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for filename in tqdm(txt_files, desc="Chunking documents"):
            path = os.path.join(INPUT_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            sents = sent_tokenize(text)
            chunks = chunk_text(sents)

            for i, chunk in enumerate(chunks):
                rec = {
                    "doc_id": f"{filename}_{i}",
                    "source_file": filename,
                    "text": chunk.strip(),
                }
                out.write(json.dumps(rec) + "\n")

    print(f"\nWrote chunks for {len(txt_files)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
