# End-to-End Graph-RAG Evaluation Report

## Executive Summary
Benchmark `prompt20-controlled-v1` compared VECTOR_RAG, GRAPH_RAG, HYBRID_RAG, and AGENTIC_HYBRID_RAG on this controlled benchmark. Results are not claims of statistical significance.

## Benchmark
- Cases: 30
- Checksum: `a06ec927804eb384588ed72ae12690cdb1c47c004f62d840415659f9812d045c`

## System Variants
- VECTOR_RAG: vector retrieval, answer generation, citation validation, grounding.
- GRAPH_RAG: benchmark-provided graph retrieval, answer generation, citation validation, grounding.
- HYBRID_RAG: benchmark-provided vector + graph retrieval with RRF, answer generation, citation validation, grounding.
- AGENTIC_HYBRID_RAG: production Prompt 19 workflow with planning, sufficiency, optional refinement, answer, citations, grounding.

## Overall Results
| System | Answer Accuracy | Correct Abstention | Citation Validity | High-Confidence Error | p95 Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| VECTOR_RAG | 0.091 | 0.000 | 1.000 | 0.933 | 31.5 ms |
| GRAPH_RAG | 0.381 | 0.000 | 1.000 | 0.467 | 310.5 ms |
| HYBRID_RAG | 0.381 | 0.000 | 1.000 | 0.529 | 63.0 ms |
| AGENTIC_HYBRID_RAG | 0.455 | 0.750 | 1.000 | 0.500 | 111.1 ms |

## Results by Category
| Category | Vector | Graph | Hybrid | Agentic |
| --- | ---: | ---: | ---: | ---: |
| semantic | 0.400 | 0.000 | 0.000 | 0.400 |
| structural | 0.000 | 0.800 | 0.800 | 0.800 |
| shared_entity | 0.000 | 0.250 | 0.250 | 0.250 |
| multi_hop | 0.000 | 0.750 | 0.750 | 0.750 |
| mixed | 0.000 | 0.000 | 0.000 | 0.000 |
| unanswerable | 0.000 | 0.000 | 0.000 | 0.000 |
| ambiguous | 0.000 | 0.000 | 0.000 | 0.000 |

## Vector RAG
- Cases: 30
- Answer accuracy: 0.091
- Evidence recall: 0.077
- Grounded answer rate: 1.000
- LLM calls/query: 1.00

## Graph RAG
- Cases: 30
- Answer accuracy: 0.381
- Evidence recall: 0.587
- Grounded answer rate: 0.714
- LLM calls/query: 0.50

## Hybrid RAG
- Cases: 30
- Answer accuracy: 0.381
- Evidence recall: 0.587
- Grounded answer rate: 0.810
- LLM calls/query: 0.57

## Agentic Hybrid Graph-RAG
- Cases: 30
- Answer accuracy: 0.455
- Evidence recall: 0.673
- Grounded answer rate: 0.864
- LLM calls/query: 2.10

## Abstention Performance
- VECTOR_RAG: correct abstention 0.000; false answer 1.000
- GRAPH_RAG: correct abstention 0.000; false answer 0.000
- HYBRID_RAG: correct abstention 0.000; false answer 0.000
- AGENTIC_HYBRID_RAG: correct abstention 0.750; false answer 0.250

## Citation Performance
- VECTOR_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000
- GRAPH_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000
- HYBRID_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000
- AGENTIC_HYBRID_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000

## Confidence Analysis
- VECTOR_RAG: {'high': 30, 'medium': 0, 'low': 0, 'insufficient_evidence': 0}; high-confidence error rate 0.933
- GRAPH_RAG: {'high': 15, 'medium': 0, 'low': 0, 'insufficient_evidence': 15}; high-confidence error rate 0.467
- HYBRID_RAG: {'high': 17, 'medium': 0, 'low': 0, 'insufficient_evidence': 13}; high-confidence error rate 0.529
- AGENTIC_HYBRID_RAG: {'high': 20, 'medium': 0, 'low': 0, 'insufficient_evidence': 10}; high-confidence error rate 0.500

## Refinement Analysis
- VECTOR_RAG: refinement rate 0.000; success rate 0.000
- GRAPH_RAG: refinement rate 0.000; success rate 0.000
- HYBRID_RAG: refinement rate 0.000; success rate 0.000
- AGENTIC_HYBRID_RAG: refinement rate 0.067; success rate 0.000

## Latency and Cost
- VECTOR_RAG: mean 13.0 ms / median 15.0 ms / p95 31.5 ms; LLM calls/query 1.00
- GRAPH_RAG: mean 103.9 ms / median 47.0 ms / p95 310.5 ms; LLM calls/query 0.50
- HYBRID_RAG: mean 34.0 ms / median 31.0 ms / p95 63.0 ms; LLM calls/query 0.57
- AGENTIC_HYBRID_RAG: mean 56.9 ms / median 47.0 ms / p95 111.1 ms; LLM calls/query 2.10

## Agentic Wins
- `ambig-dataset`
- `ambig-method`
- `ambig-paper-a`
- `ambig-task`
- `mixed-approach-datasets-a`
- `mixed-paper-c`
- `struct-datasets-a`
- `unans-missing-limitation-b`
- `unans-missing-property`
- `unans-nonexistent-relation`

## Agentic Losses
- `mixed-attack-methods-datasets`
- `mixed-refined-a`
- `multi-hop-citation-neighborhood`
- `multi-hop-methods-alt-dataset`
- `multi-hop-methods-dataset`
- `sem-approach-c`
- `sem-limit-a`
- `sem-method-a`
- `sem-problem-b`
- `sem-refined-a`
- `shared-dataset-a`
- `shared-dataset-b`
- `shared-method-a`
- `shared-task-a`
- `struct-datasets-b`
- `struct-methods-a`
- `struct-methods-c`
- `struct-tasks-a`
- `unans-missing-dataset`

## Failure Analysis
- VECTOR_RAG: {'none': 2, 'retrieval_miss': 28}
- GRAPH_RAG: {'graph_request_missing': 7, 'none': 8, 'answer_incorrect': 7, 'retrieval_miss': 2, 'execution_error': 6}
- HYBRID_RAG: {'graph_request_missing': 7, 'none': 8, 'answer_incorrect': 7, 'retrieval_miss': 2, 'execution_error': 6}
- AGENTIC_HYBRID_RAG: {'none': 10, 'retrieval_miss': 4, 'insufficient_retrieval_evidence': 5, 'answer_incorrect': 6, 'entity_not_found': 1, 'entity_ambiguous': 4}

## Limitations
Controlled benchmark results validate system mechanics and comparison logic. Fake providers do not measure live model quality, real-corpus coverage, extraction quality, or semantic claim entailment.

## Reproducibility
- Evaluation version: `v1`
- Run fingerprint: `2dfc21f984743ce452038f48a95df9508c2f2f474c192ca95bcb2a796574bc0b`
- Embedding: `sentence_transformers` / `sentence-transformers/all-MiniLM-L6-v2`
- RRF K: `60`
- Planner version: `v1`
- Critic rules: `v1`
- Answer prompt: `v1`
- Citation validator: `v1`
- Grounding rules: `v1`
- Generated at: `2026-08-19T15:51:50.698910+00:00`
