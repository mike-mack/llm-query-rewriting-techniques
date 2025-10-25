# llm-query-rewriting-techniques
Practical scripts that showcase query rewriting techniques for Retrieval-Augmented Generation (RAG). Each script is a minimal, runnable example.

## Techniques in this repo
- `multi_query_retrieval.py` — Multi-Query Retrieval. Generates alternative phrasings of a question, retrieves per query, deduplicates, and answers from combined context. Uses Chroma and OpenAI embeddings.
- `hypothetical_document_embeddings.py` — HyDE. Generates a plausible answer (a hypothetical document), embeds that, and retrieves nearest documents; then surfaces the results.
- `rewrite_retrieve_read.py` — Rewrite–Retrieve–Read. Rewrites the user query for retrieval, searches, then answers using the retrieved context.
- `rag_fusion.py` — RAG Fusion with Reciprocal Rank Fusion (RRF). Generates multiple queries, retrieves per query, fuses ranked lists via RRF, and answers from the fused context.
- `step_back.py` — Step-Back Prompting. Generalizes the original query to a broader version that can improve recall in downstream retrieval.
- `least_to_most.py` — Least-to-Most Decomposition. Breaks a complex question into simpler sub-questions to solve in order.

Notes common to the retrieval-focused scripts:
- Vector store: Chroma (no FAISS dependency required).
- Embeddings: `text-embedding-3-small` via `langchain_openai.OpenAIEmbeddings`.
- Chat model: `gpt-4o-mini` via `langchain_openai.ChatOpenAI`.
- API key is read from `OPENAI_API_KEY` (env or `.env` file). Scripts fail fast with a clear message if missing.

## Quickstart

Prereqs:
- Python 3.11+
- An OpenAI API key in your environment (or a local `.env`)

1) Create a `.env` file with your key:

```
OPENAI_API_KEY=sk-...
```

2) Create a virtualenv and install deps:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3) Run a technique script:

```
# Multi-Query Retrieval
python multi_query_retrieval.py

# HyDE (Hypothetical Document Embeddings)
python hypothetical_document_embeddings.py

# Rewrite–Retrieve–Read
python rewrite_retrieve_read.py

# RAG Fusion (Reciprocal Rank Fusion)
python rag_fusion.py

# Step-Back Prompting
python step_back.py

# Least-to-Most Decomposition
python least_to_most.py
```