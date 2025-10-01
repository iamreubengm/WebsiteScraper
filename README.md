# WebsiteScraper

Steps:
```python -m src.scrape.crawl
python -m src.scrape.parse_pdf
python -m src.preprocess.clean
python -m src.preprocess.dedupe
```

Final Documents in: data_clean/documents.dedup.jsonl