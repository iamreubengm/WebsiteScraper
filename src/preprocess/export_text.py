# -*- coding: utf-8 -*-
"""
Export plain text from documents.jsonl/documents.dedup.jsonl.

Outputs:
- data_clean/text_corpus/by_doc/<safe_doc_id>.txt   (one file per doc)
- data_clean/text_corpus/corpus.txt                 (all docs concatenated)
- data_clean/text_corpus/manifest.tsv               (index for auditing)

Usage:
  python -m src.preprocess.export_text \
      --in data_clean/documents.dedup.jsonl \
      --outdir data_clean/text_corpus \
      --min_chars 100
"""
import os, re, json, argparse
from pathlib import Path

SEP = "\n\n" + ("-" * 80) + "\n\n"

def slugify(s: str) -> str:
    # keep alnum, dash, underscore; collapse runs; cap length for Windows
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:200] if s else "doc"

def combine_text(doc: dict, include_title=True, include_headings=True) -> str:
    parts = []
    if include_title and doc.get("title"):
        parts.append(str(doc["title"]).strip())
    for sec in doc.get("sections", []):
        h = sec.get("heading") or ""
        t = sec.get("text") or ""
        if include_headings and h and h != "##":
            parts.append(str(h).strip())
        if t:
            parts.append(str(t).strip())
    # squeeze excessive blank lines
    txt = "\n\n".join(p for p in parts if p)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to documents.jsonl or documents.dedup.jsonl")
    ap.add_argument("--outdir", default="data_clean/text_corpus",
                    help="Directory for outputs")
    ap.add_argument("--min_chars", type=int, default=0,
                    help="Skip docs whose combined text is shorter than this")
    ap.add_argument("--no_titles", action="store_true",
                    help="Do not prepend titles")
    ap.add_argument("--no_headings", action="store_true",
                    help="Do not include section headings")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    outdir = Path(args.outdir)
    by_doc_dir = outdir / "by_doc"
    outdir.mkdir(parents=True, exist_ok=True)
    by_doc_dir.mkdir(parents=True, exist_ok=True)

    total, kept = 0, 0
    corpus_lines = []
    manifest_rows = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            txt = combine_text(
                doc,
                include_title=not args.no_titles,
                include_headings=not args.no_headings
            )
            if len(txt) < args.min_chars:
                continue

            # File per doc
            doc_id = (doc.get("doc_id") or "doc").replace("sha1:", "")
            fname = slugify(doc_id) + ".txt"
            out_path = by_doc_dir / fname
            out_path.write_text(txt, encoding="utf-8", newline="\n")

            # Add to big corpus with a readable header
            header = f'<<<DOC id="{doc.get("doc_id","")}" url="{doc.get("source_url","")}" title="{(doc.get("title") or "").replace(chr(34), "")}">>>'
            corpus_lines.append(header + "\n" + txt + "\n")

            # Manifest row
            manifest_rows.append("\t".join([
                fname,
                doc.get("doc_id",""),
                doc.get("source_url",""),
                (doc.get("title") or "").replace("\t", " ").strip(),
                str(len(txt))
            ]))

            kept += 1

    # Write combined corpus + manifest
    (outdir / "corpus.txt").write_text(SEP.join(corpus_lines), encoding="utf-8", newline="\n")
    (outdir / "manifest.tsv").write_text(
        "filename\tdoc_id\tsource_url\ttitle\tchars\n" + "\n".join(manifest_rows),
        encoding="utf-8",
        newline="\n"
    )

    print(f"[export_text] processed={total} kept={kept} -> {outdir}")
    print(f"[export_text] per-doc: {by_doc_dir}")
    print(f"[export_text] corpus:  {outdir / 'corpus.txt'}")
    print(f"[export_text] manifest:{outdir / 'manifest.tsv'}")

if __name__ == "__main__":
    main()
