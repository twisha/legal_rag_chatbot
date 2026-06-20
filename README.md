# Legal & Regulatory RAG Chatbot

A retrieval-augmented generation (RAG) chatbot for querying US federal regulatory documents. Built with Claude, ChromaDB, and Streamlit.

Covers documents from:

- **EPA** — Environmental Protection Agency
- **FTC** — Federal Trade Commission
- **CFPB** — Consumer Financial Protection Bureau
- **SEC** — Securities and Exchange Commission

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file:

```text
ANTHROPIC_API_KEY=your_key_here
```

### 3. Download regulatory documents

```bash
python download_legal_data.py
```

Fetches the latest rules, proposed rules, and notices from the Federal Register API (no API key required).

### 4. Build the vector index

```bash
python ingest.py
```

Chunks documents and stores embeddings in ChromaDB (`chroma_legal/`).

### 5. Run the app

```bash
streamlit run app.py
```

## How it works

1. `download_legal_data.py` fetches documents from the [Federal Register API](https://www.federalregister.gov/developers/api/v1) and saves them as text files in `data_legal/`.
2. `ingest.py` splits documents into chunks, embeds them with `thenlper/gte-small`, and stores them in a local ChromaDB index.
3. `app.py` retrieves the top-5 relevant chunks for each query and streams an answer from Claude (`claude-opus-4-8`).

## Sample prompts

### EPA

- What are the latest EPA rules on air quality standards?
- Summarize recent EPA proposed rules on greenhouse gas emissions.
- What enforcement actions has the EPA taken regarding water pollution?

### FTC

- What does the FTC require for subscription cancellation ("click to cancel")?
- Summarize the FTC's rules on dark patterns in e-commerce.
- What are the FTC's guidelines on endorsements and testimonials?

### CFPB

- What are the CFPB's rules on mortgage servicing?
- How does the CFPB regulate payday lending?
- What disclosures are required under CFPB credit card rules?

### SEC

- What are the SEC's recent rules on climate-related disclosures?
- Summarize SEC rules on insider trading reporting requirements.
- What does the SEC require for SPACs under its latest proposed rules?

## Notes

- Answers are grounded strictly in retrieved documents — the model will say so if information is not available.
- This tool provides informational research only, not legal advice.
