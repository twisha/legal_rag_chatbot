"""
app_legal.py — Legal & Regulatory RAG chatbot using Claude + ChromaDB.

Usage:
    streamlit run app_legal.py

Required:
    ANTHROPIC_API_KEY set in environment or .env file
"""

import os
import anthropic
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_legal"
EMBED_MODEL = "thenlper/gte-small"
CLAUDE_MODEL = "claude-opus-4-8"
TOP_K = 5

SYSTEM_PROMPT = """You are a Legal & Regulatory Research Assistant specializing in US federal regulations.

You have access to documents from the EPA (environmental rules), FTC (consumer protection),
CFPB (financial protection), and SEC (securities regulations).

Rules:
1. Answer ONLY from the provided CONTEXT documents.
2. If the answer is not in the context, say: "I don't have that information in the retrieved documents."
3. Cite the document title and agency when possible.
4. Be accurate and concise.
5. Note: this is informational only, not legal advice."""

CONTEXT_TEMPLATE = """CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{question}

Answer based on the context documents above."""


@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Doc {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def stream_answer(question: str, retriever):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = CONTEXT_TEMPLATE.format(context=context, question=question)

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# --- UI ---

st.set_page_config(page_title="Legal & Regulatory Research", page_icon="⚖️")
st.title("⚖️ Legal & Regulatory Research Assistant")
st.caption("Federal Register documents · EPA · FTC · CFPB · SEC · Powered by Claude")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY not found.\n\n"
        "Create a `.env` file with:\n```\nANTHROPIC_API_KEY=your_key_here\n```"
    )
    st.stop()

try:
    retriever = load_retriever()
except Exception as e:
    st.error(
        f"Could not load ChromaDB index: {e}\n\n"
        "Run these first:\n"
        "```\npython download_legal_data.py\npython ingest_legal.py\n```"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about EPA rules, FTC guidelines, CFPB policies, SEC regulations..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(user_input, retriever))

    st.session_state.messages.append({"role": "assistant", "content": response})
