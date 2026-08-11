# News Channel AI

A RAG-powered internal knowledge assistant, news summarization pipeline, and on-air presenter script generator for a regional news broadcaster — reconstructed from a system I designed and built at a previous employer, made runnable as a standalone project.

Runs fully offline out of the box (TF-IDF embeddings + extractive generation, no API key required) and upgrades to real embeddings and LLM generation the moment you add a key.

## Why this exists

"I built a RAG system" is one of the most common lines on an AI engineer's resume and one of the least differentiated — most candidates can describe the concept but not defend the specific choices. This project exists so I can point at a real, runnable implementation and answer the follow-up questions that actually separate a working RAG demo from a production-grade one: how do you know retrieval is working, why hybrid search instead of pure vector search, how do chunks get created, what happens when the model isn't grounded in the retrieved context.

## Architecture

```
                    ┌─────────────────────┐
                    │   Ingestion          │
                    │   (ingestion.py)     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                  ▼
    ┌─────────────────┐               ┌─────────────────────┐
    │  Chunk articles   │               │  Fit embeddings       │
    │  (400 chars,       │──────────▶ │  (TF-IDF local, or    │
    │   50 overlap)      │               │   OpenAI/Cohere API)  │
    └─────────────────┘               └──────────┬───────────┘
                                                  │
                                       ┌──────────▼───────────┐
                                       │   ChromaDB store       │  (vectors)
                                       │   + BM25 corpus        │  (keywords)
                                       └──────────┬───────────┘
                                                  │
                    ┌─────────────────────────────┴─────────────────────┐
                    ▼                                                    ▼
          ┌───────────────────┐                              ┌───────────────────┐
          │  Hybrid Retrieval    │                              │  RAG Generation      │
          │  (retrieval.py)       │─────────────────────────▶ │  (rag_chain.py)      │
          │  vector + BM25,       │       retrieved chunks      │  + faithfulness      │
          │  fused via RRF        │                              │    check              │
          └───────────────────┘                              └───────────────────┘

          ┌────────────────────────────────────────────────────────────┐
          │                    summarization.py                          │
          │   Article ──▶ 3-sentence broadcast summary                  │
          │   Article ──▶ presenter script (targeted to on-air duration) │
          └────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
git clone <your-repo-url>
cd news-channel-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional — enables real embeddings + live LLM generation
cp .env.example .env
# then edit .env and add an API key

# Run the full pipeline: ingest -> RAG queries -> summary -> presenter script
python run_demo.py

# Or start the API
python -m src.news_channel.api
# curl -X POST localhost:5001/query -H "Content-Type: application/json" -d '{"query":"Is there a flood risk in Nagpur?"}'

# Run tests
pytest tests/ -v
```

## Project structure

```
news-channel-ai/
├── src/news_channel/
│   ├── schemas.py        # Pydantic models for every pipeline stage
│   ├── embeddings.py      # Provider-agnostic embedding client (API / local TF-IDF)
│   ├── llm_client.py      # Provider-agnostic generation client (API / local extractive)
│   ├── ingestion.py        # Chunking, embedding, and ChromaDB storage
│   ├── retrieval.py        # Hybrid vector + BM25 search with RRF fusion
│   ├── rag_chain.py        # Retrieval -> generation -> faithfulness check
│   ├── summarization.py    # Broadcast summary + presenter script generation
│   └── api.py               # Flask endpoints
├── data/sample_articles/articles.json   # 6 sample regional news articles
├── tests/test_pipeline.py
└── run_demo.py
```

## Design decisions worth knowing for a technical interview

**Why hybrid search instead of vector-only retrieval.** Pure semantic search can miss exact matches that matter in news — a query for a specific tournament name should reliably surface the article containing that exact phrase, not just semantically-adjacent content. BM25 (keyword) and vector search are fused with **reciprocal rank fusion**, which combines two ranked lists using rank position rather than raw scores — necessary because a cosine-similarity scale and a BM25 scale aren't directly comparable.

**Why the embedding layer isn't a stub.** The offline fallback fits a real `TfidfVectorizer` on the ingested corpus and L2-normalizes the output so cosine similarity is meaningful — it's a genuinely weaker retriever than a trained neural embedding model, but it's real, not faked, and the interface is identical to the API-backed path. This is the same pattern as the LLM client's local-mock mode: **swap the backend, not the calling code.**

**Why the RAG chain includes a faithfulness check at all.** Knowing a RAG system is *working* is a harder problem than building one. The lexical-overlap check here is a lightweight proxy for what a proper RAGAS faithfulness score would give — it flags the clearest failure mode (the model answering from general/parametric knowledge instead of the retrieved context) without requiring a second LLM call to evaluate the first. See "what I'd add" below for the production version of this.

**Chunking parameters.** 400 characters with 50-character overlap — small enough to keep retrieved context focused, with enough overlap that a fact sitting near a chunk boundary doesn't get split away from its surrounding context. A production system would chunk by token count using the target model's tokenizer rather than raw character count; this is a documented simplification.

## What I'd add for a real production deployment

- **A proper vector database at scale** (Pinecone, Weaviate) rather than local ChromaDB — this repo's persistent Chroma store is appropriately sized for a demo corpus, not a production news archive.
- **A real neural embedding model** (`text-embedding-3-small`, Cohere `embed-v3`) as the default rather than the TF-IDF fallback — already wired in as the API path, just needs a key.
- **RAGAS evaluation harness** with a golden set of 50-100 known queries, run as a regression check before any change to chunking, embedding model, or prompts — this is the single highest-leverage thing missing from the original build, in hindsight.
- **A cross-encoder reranker** on top of the current RRF-fused hybrid results, for a second precision pass on the top-20 candidates before generation.
- **LangSmith or Langfuse tracing** on every RAG call, to debug retrieval/generation failures without re-running the pipeline manually.

## License

MIT