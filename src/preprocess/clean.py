# src/preprocess/clean.py
import os, csv, json, re
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDoc
from dateutil import parser as dtparser
from dateparser.search import search_dates
from urllib.parse import urlparse

RAW_DIR = "data_raw"
OUT_PATH = "data_clean/documents.jsonl"

DATE_PAT = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
                      r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
                      r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b', re.I)

def extract_canonical_date(text):
    # Try explicit Month Day, Year first
    m = DATE_PAT.search(text)
    if m:
        try:
            return dtparser.parse(m.group(0)).date().isoformat()
        except Exception:
            pass
    # Fallback: fuzzy parse the first reasonable date
    d = search_dates(text, languages=["en"])
    if d:
        try:
            return d[0][1].date().isoformat()
        except Exception:
            return None
    return None

def html_to_sections(html):
    # Use readability to get main content; fallback to raw
    try:
        doc = ReadabilityDoc(html)
        title = doc.short_title()
        content_html = doc.summary(html_partial=True)
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.text.strip() if soup.title else ""
        content_html = str(soup)

    soup = BeautifulSoup(content_html, "lxml")
    # Remove nav/footer common noise
    for sel in ["nav", "footer", "script", "style", "noscript", "form"]:
        for tag in soup.select(sel):
            tag.decompose()

    # Build sections using headings as boundaries
    sections = []
    cur_head = "##"
    cur_text = []
    for el in soup.descendants:
        if el.name and re.fullmatch(r'h[1-4]', el.name):
            # flush
            if cur_text:
                sections.append({"heading": cur_head, "text": "\n".join(cur_text).strip()})
                cur_text = []
            cur_head = "## " + " ".join(el.get_text(" ").split())
        elif getattr(el, "name", None) in ("p","li"):
            txt = " ".join(el.get_text(" ").split())
            if txt:
                cur_text.append(txt)
    if cur_text:
        sections.append({"heading": cur_head, "text": "\n".join(cur_text).strip()})
    return title, sections

def process_html_domain(domain):
    index_path = os.path.join(RAW_DIR, domain, "index.tsv")
    if not os.path.exists(index_path):
        return []
    docs = []
    with open(index_path, encoding="utf-8") as f:
        header = next(f)
        for line in f:
            url, h, ctype, retrieved_at = line.rstrip("\n").split("\t")
            if ctype != "text/html": 
                continue
            path = os.path.join(RAW_DIR, domain, f"{h}.html")
            if not os.path.exists(path): 
                continue
            with open(path, "rb") as fh:
                html = fh.read().decode("utf-8", errors="ignore")
            title, sections = html_to_sections(html)
            full_text = "\n\n".join(s["text"] for s in sections)
            doc = {
                "doc_id": f"sha1:{h}",
                "source_url": url,
                "title": title or urlparse(url).path.strip("/") or domain,
                "retrieved_at": retrieved_at,
                "content_type": "text/html",
                "canonical_date": extract_canonical_date(full_text),
                "sections": sections,
                "raw_path": path,
                "notes": {"language": "en", "status": "ok", "from_pdf": False}
            }
            docs.append(doc)
    return docs

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    count = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for domain in os.listdir(RAW_DIR):
            docs = process_html_domain(domain)
            for d in docs:
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                count += 1
    print(f"[clean] wrote {count} documents to {OUT_PATH}")

if __name__ == "__main__":
    main()
