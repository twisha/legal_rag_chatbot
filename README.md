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

- What are the EPA's rules on disposal of coal combustion residuals from electric utilities?
- Summarize the EPA's ozone reclassification State Implementation Plan rule.
- What pesticide tolerances has the EPA recently approved or proposed?
- What is the EPA's risk evaluation for TBBPA under TSCA?

### FTC

- What are the terms of the FTC's proposed consent order with Ascension Health Alliance?
- Summarize the FTC's proposed consent order with CMG Media Corporation.
- What does the FTC's premerger notification early termination process involve?

### CFPB

- What changes has the CFPB proposed to the Equal Credit Opportunity Act (Regulation B)?
- What is the CFPB's position on ability to repay and immigration status?
- Summarize the CFPB's rules on small business lending under ECOA.

### SEC

- What rule changes has the Long-Term Stock Exchange proposed for market makers?
- How is MIAX proposing to change fees for the Trade-by-Trade Report?
- What are Nasdaq's proposed changes to the Options Regulatory Fee?

## Notes

- Answers are grounded strictly in retrieved documents — the model will say so if information is not available.
- This tool provides informational research only, not legal advice.
