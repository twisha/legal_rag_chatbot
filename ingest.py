"""
ingest.py — Build pgvector index from legal documents.
Run once after download_legal_data.py.

Usage:
    python ingest.py
"""

import os
import psycopg
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = "data_legal"
COLLECTION_NAME = "legal_docs"
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
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment or .env file")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Building pgvector index (this takes a minute)...")
    vector_store = PGVector.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection=db_url,
        pre_delete_collection=True,
    )
    print(f"pgvector index saved to collection '{COLLECTION_NAME}'")
    return vector_store


def create_ivfflat_index(db_url: str):
    # Convert SQLAlchemy URL scheme to plain psycopg scheme
    conn_str = db_url.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(conn_str) as conn:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_legal_embedding
            ON langchain_pg_embedding
            USING ivfflat ((embedding::vector(384)) vector_cosine_ops)
            WITH (lists = 100)
        """)
        conn.commit()
    print("IVFFlat index created on embedding column")


if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment or .env file")
    docs = load_docs(DATA_DIR)
    chunks = chunk_docs(docs)
    build_index(chunks)
    create_ivfflat_index(db_url)
    print("\nDone! Next: streamlit run app.py")
