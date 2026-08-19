# ArXiv Hybrid Graph-RAG Platform

## Current status

This repository currently contains the **backend foundation, a domain
layer, a PostgreSQL metadata/ingestion control plane, arXiv paper
discovery, raw PDF acquisition, scientific PDF parsing, section-aware
chunking, local embeddings, Qdrant vector indexing/search, scientific
entity/relationship extraction into a validated, provenance-preserving
graph artifact, deterministic canonical entity resolution, and Neo4j
knowledge graph indexing/retrieval, plus unified evidence normalization
and cross-store provenance resolution, explicit hybrid retrieval with
RRF evidence fusion, deterministic retrieval evaluation and controlled
end-to-end answer evaluation,
natural-language query analysis/retrieval planning, and bounded
LangGraph retrieval orchestration with evidence sufficiency assessment
and one targeted retrieval refinement round, plus grounded answer
generation over a closed evidence pool with deterministic citation
validation, final grounding gates, confidence classification, and safe
abstention**: a FastAPI service with configuration, logging, exception
handling, tests, Docker support, stable domain models/identifiers
(`app/domain/`), PostgreSQL persistence (`app/storage/postgres/`), arXiv
search + normalization (`app/ingestion/discovery/`), explicit PDF
download + local storage (`app/ingestion/download/`), page-aware PDF
parsing + section recovery (`app/ingestion/parsing/`), deterministic
section-aware chunking (`app/ingestion/chunking/`), a local
`sentence-transformers` embedding provider (`app/embeddings/`), a Qdrant
vector store adapter (`app/storage/qdrant/`), vector indexing
orchestration (`app/ingestion/vector_indexing/`), deterministic semantic
search (`app/retrieval/`), an OpenAI-compatible LLM provider (`app/llm/`),
deterministic + LLM-assisted scientific knowledge extraction
(`app/ingestion/graph_extraction/`), conservative canonical resolution
(`app/ingestion/canonical_resolution/`), Neo4j graph indexing/
inspection (`app/ingestion/graph_indexing/`, `app/graph/`), and bounded
deterministic graph retrieval (`app/retrieval/graph_search.py`), with
unified evidence adapters and a Qdrant-backed graph source-chunk bridge
(`app/retrieval/evidence.py`), explicit-strategy hybrid retrieval
(`app/retrieval/hybrid.py`), deterministic retrieval evaluation
and controlled end-to-end answer evaluation (`evaluation/`), bounded
query planning (`app/retrieval/planning.py`), and a bounded self-correcting LangGraph retrieval workflow
(`app/retrieval/workflow.py`) with grounded answer generation
(`app/generation/`).

Current implemented capability:

- arXiv discovery + metadata persistence
- explicit PDF acquisition with checksum validation:
  `POST /api/v1/papers/{paper_id}/ingest`
- scientific PDF parsing + section recovery:
  `POST /api/v1/papers/{paper_id}/parse`
- parsed-document inspection: `GET /api/v1/papers/{paper_id}/document`
- deterministic section-aware chunking:
  `POST /api/v1/papers/{paper_id}/chunk`
- chunk inspection: `GET /api/v1/papers/{paper_id}/chunks`
- local embedding + Qdrant vector indexing:
  `POST /api/v1/papers/{paper_id}/vector-index`
- deterministic semantic vector search (not RAG -- no LLM involved):
  `POST /api/v1/search/vector`
- scientific entity/relationship extraction (deterministic Paper/Author/
  citation facts + LLM-assisted Method/Dataset/Task extraction, validated
  and provenance-preserving):
  `POST /api/v1/papers/{paper_id}/extract-graph`
- graph extraction inspection: `GET /api/v1/papers/{paper_id}/graph-extraction`
- deterministic canonical entity resolution + Neo4j graph indexing:
  `POST /api/v1/papers/{paper_id}/graph-index`
- graph inspection:
  `GET /api/v1/graph/papers/{paper_id}` and
  `GET /api/v1/graph/entities/{entity_id}`
- deterministic Neo4j graph retrieval (one-hop and bounded multi-hop,
  not hybrid retrieval): `POST /api/v1/search/graph`
- unified evidence objects for vector and graph search responses,
  preserving source-store identity, score semantics, stable evidence IDs,
  and graph-to-Qdrant source chunk links
- explicit vector/graph/hybrid retrieval orchestration with RRF evidence
  fusion and closed evidence-pool labels:
  `POST /api/v1/search/retrieve`
- deterministic retrieval evaluation for VECTOR vs GRAPH vs HYBRID:
  `python scripts/evaluate_retrieval.py`
- controlled end-to-end answer evaluation for VECTOR_RAG vs GRAPH_RAG vs
  HYBRID_RAG vs AGENTIC_HYBRID_RAG:
  `python scripts/evaluate_end_to_end.py`
- natural-language query intent analysis and automatic VECTOR / GRAPH /
  HYBRID retrieval-plan construction without executing retrieval:
  `POST /api/v1/query/plan`
- deterministic graph operation planning and entity resolution for
  supported graph intents
- LangGraph-based bounded retrieval orchestration with automatic
  VECTOR / GRAPH / HYBRID execution and provenance-preserving evidence
  pools, evidence sufficiency assessment, and at most one targeted
  refinement: `POST /api/v1/query/retrieve`
- grounded answer generation over the final closed evidence pool, with
  deterministic validation of inline evidence markers into trusted
  citations, rule-based final confidence, and deterministic abstention:
  `POST /api/v1/query/answer`
- ingestion/parse/chunk/vector-index/extraction/graph-index-state tracking:
  `GET /api/v1/ingestion/{ingestion_job_id}`

**Not implemented**: live-model end-to-end evaluation, real-corpus
portfolio evaluation, semantic claim-to-citation entailment verification,
conversation memory, or multi-agent orchestration.
Canonical entity resolution is intentionally conservative: exact trusted
paper identity, exact normalized names, and explicit aliases only. There
is no fuzzy merging, no LLM canonicalization, and no claim that differently
named concepts such as `MIMIC` and `MIMIC-IV` are equivalent. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full discovery/
acquisition/parsing/chunking/vector-indexing/extraction distinction and
what's implemented vs. planned overall.

## Local setup

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Install dependencies (including test tools):

```bash
pip install -e ".[dev]"
```

Copy the example environment file and adjust as needed:

```bash
cp .env.example .env
```

### Database (PostgreSQL)

Start PostgreSQL (via Docker Compose, or point `DATABASE_URL` at your own
instance):

```bash
docker compose up -d postgres
```

Apply migrations:

```bash
alembic upgrade head
```

Generate a new migration after changing `app/storage/postgres/models.py`:

```bash
alembic revision --autogenerate -m "describe the change"
```

Always review an autogenerated migration before applying it. Never resolve
a schema mismatch by deleting/recreating the database — add a migration.

**Applied migrations are immutable history.** Once a migration has run
anywhere (including your own local dev database), never edit or regenerate
that revision file to "fix" it — create a new, additive migration instead.
Rewriting an already-applied revision desyncs any database that already
recorded it in `alembic_version` (an early snapshot of this project had to
recover from exactly that after a since-corrected development mistake).

### Tests

```bash
pytest
```

Domain-layer and API tests always run. PostgreSQL/Qdrant integration tests
(`tests/integration/`) automatically **skip** unless a real database/Qdrant
is reachable at `DATABASE_URL`/`QDRANT_URL` (or `TEST_DATABASE_URL`/
`TEST_QDRANT_URL`, if you want to point tests at different instances than
your dev ones), e.g.:

```bash
TEST_DATABASE_URL=postgresql+psycopg://arxiv:arxiv@localhost:5432/arxiv_graph_rag \
TEST_QDRANT_URL=http://localhost:6333 \
pytest
```

Vector-indexing/search tests never download a real embedding model --
they use a deterministic fake `EmbeddingProvider` (`tests/embeddings/fakes.py`).
Graph-extraction tests never call a real LLM -- they use a deterministic
fake `LLMProvider` (`tests/llm/fakes.py`).

### Retrieval Evaluation

Run the deterministic VECTOR vs GRAPH vs HYBRID benchmark:

```bash
python scripts/evaluate_retrieval.py
```

The default benchmark file is `evaluation/retrieval_benchmark.json`, and
reports are written to `evaluation/results/retrieval_report.json` and
`evaluation/results/retrieval_report.md`. The runner is read-only: it does
not ingest papers, download PDFs, parse, chunk, embed, reindex Qdrant, or
mutate Neo4j. CI correctness uses controlled Qdrant and Neo4j fixtures;
real-corpus evaluation should only be reported when a sufficient indexed
local corpus already exists.

Run the controlled end-to-end answer benchmark:

```bash
python scripts/evaluate_end_to_end.py \
  --database-url postgresql+psycopg://arxiv:arxiv@localhost:5432/arxiv_graph_rag \
  --qdrant-url http://localhost:6333 \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-username neo4j \
  --neo4j-password change-me
```

The default benchmark file is `evaluation/end_to_end_benchmark.json`, and
reports are written to `evaluation/results/end_to_end_report.json` and
`evaluation/results/end_to_end_report.md`. This controlled runner verifies
PostgreSQL, Qdrant, and Neo4j connectivity, seeds disposable controlled
Qdrant/Neo4j fixtures, compares `VECTOR_RAG`, `GRAPH_RAG`, `HYBRID_RAG`,
and `AGENTIC_HYBRID_RAG`, then cleans up controlled data. It uses
deterministic fake embedding/planner/critic/answer providers, so it
validates software behavior rather than live model quality.

### Query Planning

Plan retrieval from a natural-language scientific question:

```bash
curl -X POST http://localhost:8000/api/v1/query/plan \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Which datasets does GraphSteal evaluate on?\"}"
```

The planner uses the configured `LLMProvider` only for structured intent
and entity mention extraction. Application code then deterministically
enforces the VECTOR / GRAPH / HYBRID strategy, graph-operation allowlist,
entity resolution, and safe retrieval-count defaults. It does not answer
questions, execute retrieval, generate Cypher, or run LangGraph.

Execute one bounded LangGraph retrieval workflow:

```bash
curl -X POST http://localhost:8000/api/v1/query/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Which datasets does GraphSteal evaluate on?\"}"
```

This endpoint runs query analysis, deterministic planning, initial
retrieval, evidence-pool construction, sufficiency assessment, and, when
needed, one bounded targeted refinement using the existing retrieval
services. Structural graph sufficiency is deterministic where reliable;
semantic and mixed evidence gaps may use the configured LLM provider as a
critic. It still returns evidence only, not a final synthesized answer.

Generate a grounded answer from the final closed evidence pool:

```bash
curl -X POST http://localhost:8000/api/v1/query/answer \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Explain GraphSteal's approach and list its datasets.\"}"
```

This endpoint reuses the bounded retrieval workflow, generates only when
the final evidence assessment is sufficient, and supplies the answer model
only the final closed `E1`, `E2`, ... evidence pool. Generated markers are
then parsed deterministically, validated against the actual answer context
and final evidence pool, renumbered as user-facing `[1]`, `[2]`, ...
citations, and mapped to trusted citation objects built only from real
retrieved evidence. A final deterministic grounding decision then returns
either the sanitized validated answer with categorical confidence
(`high`, `medium`, `low`) or a fixed safe abstention response with
`insufficient_evidence`. This validates citation identity and final
grounding gates, not semantic entailment of every uncited claim.

Run the deterministic fixture-based planner validation:

```bash
python scripts/evaluate_query_planning.py
```

The report is written to `evaluation/results/planning_report.json`. This
validates planner software behavior; it does not measure live LLM
classification accuracy.

### Run the API locally

```bash
uvicorn app.main:app --reload
```

Then check:

```text
GET  http://localhost:8000/api/v1/health                       # liveness — never depends on the database
GET  http://localhost:8000/api/v1/health/db                    # readiness — checks PostgreSQL connectivity
GET  http://localhost:8000/api/v1/papers/search?q=graph+rag    # search arXiv + persist metadata
POST http://localhost:8000/api/v1/papers/{paper_id}/ingest     # explicit PDF download + storage
POST http://localhost:8000/api/v1/papers/{paper_id}/parse      # explicit PDF parsing + section recovery
GET  http://localhost:8000/api/v1/papers/{paper_id}/document   # inspect the parsed structure
POST http://localhost:8000/api/v1/papers/{paper_id}/chunk      # explicit section-aware chunking
GET  http://localhost:8000/api/v1/papers/{paper_id}/chunks     # inspect the chunk corpus
POST http://localhost:8000/api/v1/papers/{paper_id}/vector-index # explicit embedding + Qdrant indexing
POST http://localhost:8000/api/v1/search/vector                # semantic search (not RAG -- no LLM)
POST http://localhost:8000/api/v1/query/plan                   # plan retrieval from natural language
POST http://localhost:8000/api/v1/query/retrieve               # one bounded LangGraph retrieval workflow
POST http://localhost:8000/api/v1/query/answer                 # grounded answer over closed evidence pool
GET  http://localhost:8000/api/v1/health/qdrant                 # Qdrant readiness
GET  http://localhost:8000/api/v1/health/neo4j                  # Neo4j readiness
POST http://localhost:8000/api/v1/papers/{paper_id}/extract-graph      # explicit scientific knowledge extraction
GET  http://localhost:8000/api/v1/papers/{paper_id}/graph-extraction   # inspect the extraction artifact
POST http://localhost:8000/api/v1/papers/{paper_id}/graph-index         # canonicalize + write to Neo4j
GET  http://localhost:8000/api/v1/graph/papers/{paper_id}               # inspect one paper's graph
GET  http://localhost:8000/api/v1/graph/entities/{entity_id}            # inspect one canonical entity
POST http://localhost:8000/api/v1/search/graph                          # deterministic graph retrieval
GET  http://localhost:8000/api/v1/ingestion/{ingestion_job_id} # ingestion/parse/chunk/vector-index/extraction job status
```

## Frontend

The portfolio UI lives in `frontend/` and calls the real FastAPI answer endpoint.

```bash
cd frontend
npm install
npm run dev
```

Configuration:

```text
VITE_API_BASE_URL=http://localhost:8000
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Local ports:

```text
FastAPI: http://localhost:8000 or http://127.0.0.1:8000
React:   http://localhost:5173 or http://127.0.0.1:5173
```

The interface shows the final validated answer, categorical confidence,
trusted citations, selected retrieval strategy, graph operation, evidence,
graph relationships/paths, retrieval rounds/refinement, and safe workflow
trace. Complex mixed queries involving multiple graph operations remain
limited in V1 and are shown truthfully as backend results.

`extract-graph` requires `LLM_BASE_URL` and `LLM_MODEL` to be configured
(e.g. a local LM Studio server, or any OpenAI-compatible endpoint) -- it
returns a clear `503` if the LLM endpoint is unreachable, not a silent
fallback.

### Troubleshooting: planner structured-output failure

`POST /api/v1/query/plan`, `/retrieve`, and `/answer` require the
configured LLM to return one JSON object that validates against the
planner schema. The OpenAI-compatible provider requests
`response_format: {"type": "json_object"}` and tolerates only harmless
formatting wrappers: one enclosing JSON code fence, or one balanced
top-level JSON object surrounded by short wrapper text. It does not repair
invalid enum values, convert malformed entity objects into entity lists,
or infer missing semantic fields.

If a live model returns a planner error such as
`LLM returned unparseable/invalid structured output`, verify `LLM_MODEL`,
`LLM_BASE_URL`, and `LLM_MAX_RETRIES`, then inspect the safe failure
message for the invalid field names. Valid planner intents are exact enum
values such as `semantic_explanation`, `paper_datasets`,
`shared_methods`, `datasets_from_citing_papers`, and
`mixed_semantic_structural`; arbitrary graph-operation names are rejected
before retrieval.

### Manual smoke-test scripts (live network, outside `pytest`)

```bash
python scripts/seed_demo_corpus.py                 # seed the local GraphSteal UI demo corpus
python scripts/search_arxiv.py "graph rag"           # search only, no persistence
python scripts/ingest_arxiv_pdf.py "graph rag"        # search + real PDF download (needs DATABASE_URL)
python scripts/inspect_parsed_paper.py <paper_id>     # print a parsed paper's section structure
python scripts/inspect_chunks.py <paper_id>           # print a chunked paper's chunk structure
python scripts/inspect_vector_index.py <paper_id>     # print a paper's Qdrant reconciliation state
python scripts/vector_search.py "graph neural network attack"  # real semantic search, no LLM
python scripts/inspect_graph_extraction.py <paper_id>  # print a paper's entities/relationships
python scripts/inspect_graph.py <paper_id>             # print a paper's indexed Neo4j graph
python scripts/resolve_entity.py dataset "MIMIC-IV"    # inspect deterministic canonical resolution
python scripts/graph_search.py paper-datasets <paper_id> # deterministic Neo4j graph retrieval
```

None are part of the automated test suite — use them to sanity-check the
discovery/download/parsing/chunking/vector-indexing/extraction pieces
against real arXiv independent of the API. The first real embedding call
downloads the configured `EMBEDDING_MODEL` from Hugging Face if not
already cached locally (see docs/ARCHITECTURE.md). `extract-graph`
requires a real LLM endpoint (LM Studio or OpenAI-compatible) to be
configured and reachable. `graph-index` does not call an LLM; it requires
Neo4j to be configured and reachable.

### CI/CD

GitHub Actions workflows verify backend/frontend changes, publish the
backend API image to GitHub Container Registry, and deploy the static
frontend to GitHub Pages. See [docs/CI_CD_PLAN.md](docs/CI_CD_PLAN.md).

### Run with Docker Compose

```bash
docker compose up --build
```

This starts `postgres`, `qdrant`, and `neo4j` (all with health checks and
persistent volumes) and `api` (which waits for those dependencies to be
healthy before starting). Downloaded PDFs persist across container
restarts/rebuilds via the `paper_storage` volume; indexed vectors persist
via the `qdrant_data` volume; graph data persists via the `neo4j_data`
volume. The API is available at
`http://localhost:8000`. Run `alembic upgrade head` against it separately
(Compose does not run migrations automatically at this stage).

## Planned architecture

This foundation will be extended incrementally to add:

- answer-quality evaluation
- observability

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the high-level target
architecture and what's implemented today vs. planned.
