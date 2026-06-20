"""
ingest_legal.py — Build ChromaDB vector index from legal documents.
Run once after download_legal_data.py.

Usage:
    python ingest_legal.py
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DATA_DIR = "data_legal"
CHROMA_DIR = "chroma_legal"
EMBED_MODEL = "thenlper/gte-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_docs(data_dir: str):
    paths = list(Path(data_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files in '{data_dir}'. Run download_legal_data.py first."
        )
    docs = []
    for path in paths:
        try:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    print(f"Loaded {len(docs)} documents from {len(paths)} files")
    return docs


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_index(chunks):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Building ChromaDB index (this takes a minute)...")
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DIR,
    )
    count = vector_store._collection.count()
    print(f"ChromaDB index saved to '{CHROMA_DIR}/' ({count} vectors)")


if __name__ == "__main__":
    docs = load_docs(DATA_DIR)
    chunks = chunk_docs(docs)
    build_index(chunks)
    print("\nDone! Next: streamlit run app_legal.py")
