"""
eval.py — Evaluate the RAG pipeline using Claude as judge.

Usage:
    python eval.py

Scores each test case on three dimensions (1–5):
  - Faithfulness:  every claim is supported by the retrieved chunks
  - Relevance:     the answer addresses the question
  - Groundedness:  no hallucination beyond the provided context
"""

import json
import os
import anthropic
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

COLLECTION_NAME = "legal_docs"
EMBED_MODEL = "thenlper/gte-small"
GENERATOR_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-opus-4-8"
TOP_K = 5

TEST_CASES = [
    # EPA
    {
        "agency": "EPA",
        "question": "What are the EPA's rules on disposal of coal combustion residuals from electric utilities?",
    },
    {
        "agency": "EPA",
        "question": "Summarize the EPA's ozone reclassification State Implementation Plan rule.",
    },
    {
        "agency": "EPA",
        "question": "What pesticide tolerances has the EPA recently approved or proposed?",
    },
    # FTC
    {
        "agency": "FTC",
        "question": "What are the terms of the FTC's proposed consent order with Ascension Health Alliance?",
    },
    {
        "agency": "FTC",
        "question": "What does the FTC's premerger notification early termination process involve?",
    },
    # CFPB
    {
        "agency": "CFPB",
        "question": "What changes has the CFPB proposed to the Equal Credit Opportunity Act (Regulation B)?",
    },
    {
        "agency": "CFPB",
        "question": "What is the CFPB's position on ability to repay and immigration status?",
    },
    # SEC
    {
        "agency": "SEC",
        "question": "What rule changes has the Long-Term Stock Exchange proposed for market makers?",
    },
    {
        "agency": "SEC",
        "question": "How is MIAX proposing to change fees for the Trade-by-Trade Report?",
    },
]

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

JUDGE_PROMPT = """You are evaluating a RAG chatbot answer. Score on three dimensions from 1 (poor) to 5 (excellent).

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Scoring guide:
- faithfulness: Every claim in the answer is directly supported by the retrieved context. Penalize if the answer adds facts not in the context.
- relevance: The answer actually addresses what the question is asking.
- groundedness: The answer does not hallucinate beyond the documents. Full score if the answer correctly says it lacks information when context is absent.

Return ONLY valid JSON in this exact format:
{{"faithfulness": <1-5>, "relevance": <1-5>, "groundedness": <1-5>, "reasoning": "<one sentence>"}}"""


def load_retriever():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment or .env file")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=db_url,
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Doc {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, retriever, client: anthropic.Anthropic) -> tuple[str, str]:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
    response = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, context


def judge_answer(question: str, context: str, answer: str, client: anthropic.Anthropic) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        print(f"  [warn] judge returned non-JSON: {raw[:120]}")
        return {"faithfulness": 0, "relevance": 0, "groundedness": 0, "reasoning": "parse error"}


def run_evals():
    client = anthropic.Anthropic()
    retriever = load_retriever()

    results = []
    print(f"\nRunning {len(TEST_CASES)} eval cases...\n")

    for i, case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {case['agency']}: {case['question'][:60]}...")
        answer, context = generate_answer(case["question"], retriever, client)
        scores = judge_answer(case["question"], context, answer, client)
        results.append({**case, "answer": answer, **scores})
        avg = (scores["faithfulness"] + scores["relevance"] + scores["groundedness"]) / 3
        print(f"         F={scores['faithfulness']} R={scores['relevance']} G={scores['groundedness']} avg={avg:.1f}  {scores['reasoning']}")

    print("\n" + "=" * 68)
    print(f"{'Agency':<8} {'Question':<45} {'F':>3} {'R':>3} {'G':>3} {'Avg':>5}")
    print("-" * 68)

    totals = {"faithfulness": 0, "relevance": 0, "groundedness": 0}
    for r in results:
        avg = (r["faithfulness"] + r["relevance"] + r["groundedness"]) / 3
        print(f"{r['agency']:<8} {r['question'][:44]:<45} {r['faithfulness']:>3} {r['relevance']:>3} {r['groundedness']:>3} {avg:>5.1f}")
        for k in totals:
            totals[k] += r[k]

    n = len(results)
    print("-" * 68)
    print(
        f"{'AVERAGE':<54}"
        f" {totals['faithfulness']/n:>3.1f}"
        f" {totals['relevance']/n:>3.1f}"
        f" {totals['groundedness']/n:>3.1f}"
        f" {sum(totals.values())/(3*n):>5.1f}"
    )
    print()


if __name__ == "__main__":
    run_evals()
