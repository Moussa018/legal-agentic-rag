# Legal Agentic RAG

Question answering over legal PDFs (contracts, statutes) that combines **vector + keyword search** with a **Neo4j knowledge graph**, driven by an **LLM agent** that picks its own tools.

Plain vector RAG finds passages that *look like* the question, but misses answers that require following relationships, e.g. *"Which companies are linked to ACME by a contract expiring in 2026?"* (company → contract → date). This project extracts entities and relations into a graph so those multi-hop questions can be answered with Cypher, while hybrid text retrieval handles citations and exact terms.

## The idea

Legal questions are rarely answered by one paragraph. They need **facts scattered across documents** and **links between them**. Take:

> *"Which clauses in contracts signed by an ACME representative reference GDPR?"*

Answering it means chaining three hops: person → represents → ACME, person → party to → contract, contract → contains → clause → references → GDPR. No single chunk of text contains that chain, so similarity search alone fails.

The project splits the problem in two and lets an agent combine the halves:

| Need | Best tool | Why |
| --- | --- | --- |
| Follow relationships (who, which contract, which date) | **Knowledge graph + Cypher** | A graph query walks links exactly instead of guessing from similarity |
| Find the wording and cite it (file, page) | **Vector search** | Matches meaning even when words differ |
| Find an exact token (clause 7, article L.1234, a company name) | **BM25 keyword search** | Embeddings blur exact tokens; BM25 does not |
| Decide which of the above to use, and in what order | **LLM agent** | The right strategy depends on the question, so it is chosen at run time, not hard-coded |

The three modes (`vector`, `hybrid`, `agentic`) exist so the benefit of each layer can be **measured**: vector is the baseline, hybrid adds the graph and keyword fusion, agentic adds autonomous tool use and self-correction.

## How it works

```
Ingestion (once)
  PDF ─► chunks ─┬─► Chroma (embeddings) ─► also indexed by BM25
                 └─► LLMGraphTransformer (Claude) ─► Neo4j (closed schema)

Question (each call)
  POST /ask ─► mode ─► retrieval (Chroma + BM25) and/or Cypher (Neo4j) ─► Claude ─► answer + sources
```

## Architecture in detail

### 1. Ingestion (`app/ingest.py`)

1. **Parse and chunk.** `pypdf` extracts text page by page; chunks are 1500 characters with 200 overlap. Every chunk keeps `source` (file name) and `page`, which is what makes citations possible later.
2. **Vector index.** Chunks are upserted into a persistent Chroma collection (ids like `contract.pdf:12`). Embeddings use Chroma's default local MiniLM model, so no second API key is needed.
3. **Graph extraction.** Claude reads each chunk through `LLMGraphTransformer` and emits entities and relations, restricted to the closed schema. Restricting the schema stops the model inventing new node types per document, which keeps queries predictable.
4. **Provenance.** Graph documents are written with `include_source=True`, linking each entity to the chunk it came from.

### 2. Retrieval (`app/bm25.py`)

- **Dense**: Chroma returns the nearest chunks by meaning.
- **Lexical**: BM25 ranks chunks by exact-token overlap (lowercased, accents stripped). Its index lives in memory and rebuilds when the chunk count changes.
- **Fusion**: Reciprocal Rank Fusion takes 20 candidates from each ranking and scores each chunk by the sum of `1 / (60 + rank)`. It merges rankings without calibrating two incomparable score scales.

### 3. Graph querying

Claude writes Cypher from the live graph schema (`get_graph_schema` in agent mode, injected into the prompt in hybrid mode). Entity names live in the node `id` property, matched with `toLower(...) CONTAINS`. Queries are filtered for write keywords and capped at 50 rows. In hybrid mode an invalid query silently falls back to text context. In agent mode the error is **returned to the model so it can repair the query**.

### 4. The agent loop (`app/agent.py`)

```
messages = [system prompt, question]
repeat up to 8 times:
    Claude replies
    no tool call  -> that is the answer, stop
    tool call(s)  -> run them, append results, loop
budget exhausted -> force a best-effort answer, flag truncated: true
```

The system prompt encodes the strategy: look at the schema first for relationship questions, pick the search tool that fits the query type, decompose multi-hop questions, never repeat a failed query, and answer only from tool output while citing file and page. Every tool call is logged in `steps`, so the reasoning is auditable.

### 5. Answer generation

Claude receives the retrieved passages (and graph rows in hybrid mode) and is told to answer **only from that context**, say so when the context is insufficient, and cite sources. Temperature is 0.

### Design choices

- **Closed graph schema** over free-form extraction: consistency across documents beats coverage.
- **Both stores built from the same chunks**: a graph answer can always be traced back to source text.
- **Agent over fixed pipeline for hard questions**: it can retry and branch; the fixed `hybrid` mode stays as a cheaper, deterministic option.
- **Read-only by design**: generated Cypher is never allowed to modify the graph (see roadmap for hardening).

### Three answer modes

| Mode | What it does | Use it for |
| --- | --- | --- |
| `vector` | Top-k dense retrieval from Chroma only | Baseline |
| `hybrid` | Vector + BM25 fused with Reciprocal Rank Fusion, plus one LLM-generated Cypher query, in a single prompt | Predictable cost and latency |
| `agentic` (default) | Claude loops over tools for up to 8 steps, retries failed Cypher, refines searches, returns its trace in `steps` | Multi-hop questions |

### Agent tools

| Tool | When the agent uses it |
| --- | --- |
| `get_graph_schema` | First step for relationship questions |
| `run_cypher` | Query the graph (read-only; errors are returned so the agent can fix its query) |
| `search_documents` | Default: vector + BM25 fused |
| `keyword_search` | Exact terms: clause numbers, names, statute references, amounts |
| `semantic_search` | Concepts without precise vocabulary |

### Graph schema

The schema is closed so the graph stays consistent across documents.

- Nodes: `Personne`, `Societe`, `Contrat`, `Clause`, `Date`, `Loi`
- Relations: `PARTIE_A`, `REPRESENTE`, `CONTIENT`, `EXPIRE_LE`, `SIGNE_LE`, `REFERENCE`, `FILIALE_DE`

See `backend/app/config.py` for the exact (source, relation, target) triples.

## Tech stack

FastAPI · Claude (via `langchain-anthropic`) · ChromaDB · `rank-bm25` · Neo4j 5.26 + APOC · LangChain · `pypdf`

## Project layout

```
backend/
  app/
    config.py     env vars and closed graph schema
    stores.py     factories for the LLM, Neo4j and Chroma
    ingest.py     PDF -> Chroma + Neo4j
    bm25.py       BM25 index, vector hits, RRF fusion
    qa.py         vector and hybrid modes, prompts, read-only guard
    agent.py      agentic mode (tool loop)
    main.py       FastAPI app
  eval/
    run_eval.py   compares the three modes
    questions.json
data/pdfs/        drop your PDFs here
docker-compose.yml
.env.example
```

## Getting started

**Prerequisites:** Python 3.10+, Docker, an Anthropic API key.

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY
docker compose up -d          # Neo4j, browser at http://localhost:7474

cd backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put PDFs in data/pdfs/
python -m app.ingest
uvicorn app.main:app --reload
```

### Configuration

| Variable | Default |
| --- | --- |
| `ANTHROPIC_API_KEY` | required |
| `LLM_MODEL` | `claude-sonnet-5-5` |
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `changeme123` |
| `CHROMA_DIR` | `./data/chroma` |
| `PDF_DIR` | `./data/pdfs` |

Change the default Neo4j password before exposing anything beyond localhost.

## API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which companies are linked to ACME by a contract expiring in 2026?", "mode": "agentic"}'
```

- `POST /ask` with `{"question": str, "mode": "agentic" | "hybrid" | "vector"}` (default `agentic`)
- `GET /health` returns `{"status": "ok"}`

Responses include `answer` and `mode`. `hybrid` and `vector` add `cypher`, `graph_rows` and `sources` (file, page); `agentic` adds `steps` (tool, args, truncated result) and `truncated: true` if it ran out of steps.

## Evaluation

```bash
cd backend
python -m eval.run_eval
```

Runs every question in `eval/questions.json` through all three modes and prints a score per mode. A question passes when every expected string appears in the answer (case-insensitive). The bundled file holds only two sample questions: replace it with questions and gold answers from your own corpus before trusting the numbers.

## Known limitations and roadmap

- [ ] Run generated Cypher with a read-only Neo4j account. Today only a keyword filter blocks writes, and it also rejects legitimate `SET`/`CALL`.
- [ ] Entity normalization and deduplication (e.g. "ACME SA" vs "Acme").
- [ ] A real evaluation set: 30+ multi-hop questions with reference answers.
- [ ] Skip already-ingested PDFs; re-running ingestion repeats LLM extraction for every file.
- [ ] Frontend (React): question box, answer, sources, displayed Cypher. CORS is already open for `http://localhost:5173`.
- [ ] Cost control for extraction (one LLM call per chunk): caching or a smaller model.

## Disclaimer

This is a research and portfolio project. Its answers are not legal advice.
