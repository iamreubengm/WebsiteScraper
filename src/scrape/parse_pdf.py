# src/scrape/parse_pdf.py
import pdfplumber, json, os, glob
from datetime import datetime, timezone

def pdf_to_text(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()

def main(raw_root="data_raw", out_jsonl="data_clean/documents.jsonl"):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as out:
        for domain in os.listdir(raw_root):
            idx = os.path.join(raw_root, domain, "index.tsv")
            if not os.path.exists(idx): 
                continue
            with open(idx, encoding="utf-8") as f:
                next(f)  # header
                for line in f:
                    url, h, ctype, retrieved_at = line.rstrip("\n").split("\t")
                    if ctype != "application/pdf": 
                        continue
                    pdf_path = os.path.join(raw_root, domain, f"{h}.pdf")
                    if not os.path.exists(pdf_path): 
                        continue
                    text = pdf_to_text(pdf_path)
                    if not text: 
                        continue
                    doc = {
                        "doc_id": f"sha1:{h}",
                        "source_url": url,
                        "title": os.path.basename(pdf_path),
                        "retrieved_at": retrieved_at,
                        "content_type": "application/pdf",
                        "canonical_date": None,
                        "sections": [{"heading": "## Document", "text": text}],
                        "raw_path": pdf_path,
                        "notes": {"language": "en", "status": "ok", "from_pdf": True}
                    }
                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
