import os, re, time, hashlib, json, csv
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm

RAW_DIR = "data_raw"

def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def load_whitelist(path="src/scrape/whitelist.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def in_scope(url, allowed_domains):
    netloc = urlparse(url).netloc
    return any(d in netloc for d in allowed_domains)

def is_pdf(resp: requests.Response) -> bool:
    ct = resp.headers.get("Content-Type", "")
    return "application/pdf" in ct or resp.url.lower().endswith(".pdf")

def fetch(url, delay=0.5):
    time.sleep(delay)
    headers = {"User-Agent": "CMU-ANLP-RAG-Student Crawler (+contact team email)"}
    return requests.get(url, headers=headers, timeout=15)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_index_row(index_path, row):
    new = not os.path.exists(index_path)
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["url","sha1","content_type","retrieved_at"])
        w.writerow(row)

def extract_links(base_url, html):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"): 
            continue
        yield urljoin(base_url, href)

def crawl():
    wl = load_whitelist()
    allowed = wl["allowed_domains"]
    seeds = wl["seed_urls"]
    max_depth = int(wl.get("max_depth", 2))
    delay = float(wl.get("delay_seconds", 0.5))

    seen = set()
    frontier = [(s, 0) for s in seeds]

    with tqdm(total=len(frontier), desc="Crawl seeds") as pbar:
        while frontier:
            url, depth = frontier.pop(0)
            pbar.update(1)

            if url in seen or depth > max_depth:
                continue
            seen.add(url)

            if not in_scope(url, allowed):
                continue

            try:
                resp = fetch(url, delay=delay)
            except Exception:
                continue
            if resp.status_code != 200:
                continue

            domain = urlparse(resp.url).netloc.replace(":", "_")
            out_dir = os.path.join(RAW_DIR, domain)
            ensure_dir(out_dir)

            content_bytes = resp.content
            h = sha1(content_bytes)
            ext = ".pdf" if is_pdf(resp) else ".html"
            out_path = os.path.join(out_dir, f"{h}{ext}")
            if not os.path.exists(out_path):
                with open(out_path, "wb") as f:
                    f.write(content_bytes)

            index_path = os.path.join(out_dir, "index.tsv")
            ctype = "application/pdf" if is_pdf(resp) else "text/html"
            write_index_row(index_path, [
                resp.url, h, ctype, datetime.now(timezone.utc).isoformat()
            ])

            # enqueue children if HTML
            if ctype == "text/html" and depth < max_depth:
                try:
                    links = list(extract_links(resp.url, resp.text))
                    for link in links:
                        if in_scope(link, allowed) and link not in seen:
                            frontier.append((link, depth+1))
                    pbar.total += len(links)
                except Exception:
                    pass

if __name__ == "__main__":
    crawl()
