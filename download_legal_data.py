"""
download_legal_data.py — Fetch free regulatory documents from the Federal Register API.
No API key required.

Usage:
    python download_legal_data.py
"""

import re
import time
import urllib.parse
import requests
from pathlib import Path

DATA_DIR = Path("data_legal")
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.federalregister.gov/api/v1/documents.json"

AGENCIES = [
    "environmental-protection-agency",
    "federal-trade-commission",
    "consumer-financial-protection-bureau",
    "securities-and-exchange-commission",
]

DOCS_PER_AGENCY = 15


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def fetch_full_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return strip_html(resp.text)[:6000]
    except Exception:
        return ""


def fetch_documents(agency_slug: str) -> list:
    # Build URL manually — requests encodes brackets (%5B%5D) but the
    # Federal Register API requires literal brackets in the query string.
    parts = [
        f"per_page={DOCS_PER_AGENCY}",
        "order=newest",
        f"conditions[agencies][]={urllib.parse.quote(agency_slug)}",
        "conditions[type][]=RULE",
        "conditions[type][]=PRORULE",
        "conditions[type][]=NOTICE",
        "fields[]=document_number",
        "fields[]=title",
        "fields[]=abstract",
        "fields[]=agency_names",
        "fields[]=publication_date",
        "fields[]=topics",
        "fields[]=body_html_url",
    ]
    url = f"{BASE_URL}?{'&'.join(parts)}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"  Error fetching {agency_slug}: {e}")
        return []


def save_document(doc: dict, idx: int) -> None:
    doc_num = doc.get("document_number", f"doc_{idx}").replace("/", "-")
    path = DATA_DIR / f"{idx:04d}_{doc_num}.txt"

    full_text = ""
    body_url = doc.get("body_html_url")
    if body_url:
        full_text = fetch_full_text(body_url)

    content = f"""Title: {doc.get('title', 'No title')}
Agency: {', '.join(doc.get('agency_names', []))}
Date: {doc.get('publication_date', 'Unknown')}
Document Number: {doc.get('document_number', 'Unknown')}
Topics: {', '.join(doc.get('topics', []))}
Source: Federal Register

Abstract:
{doc.get('abstract') or 'No abstract available.'}
"""
    if full_text:
        content += f"\nFull Text (excerpt):\n{full_text}"

    path.write_text(content, encoding="utf-8")


def main():
    print("Fetching regulatory documents from Federal Register API (no API key needed)...")
    print(f"Saving to: {DATA_DIR}/\n")

    all_docs = []
    for agency in AGENCIES:
        print(f"  Fetching {agency}...")
        docs = fetch_documents(agency)
        print(f"    Got {len(docs)} documents")
        all_docs.extend(docs)
        time.sleep(1)

    print(f"\nSaving {len(all_docs)} documents...")
    for i, doc in enumerate(all_docs):
        save_document(doc, i)
        time.sleep(0.3)

    print(f"\nDone! {len(all_docs)} documents saved to {DATA_DIR}/")
    print("Next: run  python ingest_legal.py")


if __name__ == "__main__":
    main()
