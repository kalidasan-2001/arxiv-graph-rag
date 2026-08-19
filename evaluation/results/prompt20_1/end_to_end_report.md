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
| VECTOR_RAG | 0.091 | 0.000 | 1.000 | 0.933 | 32.0 ms |
| GRAPH_RAG | 0.381 | 0.000 | 1.000 | 0.467 | 309.8 ms |
| HYBRID_RAG | 0.381 | 0.000 | 1.000 | 0.529 | 62.9 ms |
| AGENTIC_HYBRID_RAG | 0.455 | 1.000 | 1.000 | 0.467 | 125.0 ms |

## Prompt 20.1 Delta
| Metric | Prompt 20 | Prompt 20.1 | Delta |
| --- | ---: | ---: | ---: |
| answer_accuracy | 0.455 | 0.455 | +0.000 |
| correct_abstention_rate | 0.750 | 1.000 | +0.250 |
| false_answer_rate | 0.250 | 0.000 | -0.250 |
| evidence_recall | 0.673 | 0.635 | -0.038 |
| refinement_rate | 0.067 | 0.100 | +0.033 |
| refinement_success_rate | 0.000 | 0.667 | +0.667 |
| HIGH_count | 20 | 15 | -5 |
| high_confidence_error_rate | 0.500 | 0.467 | -0.033 |
| p95_latency_ms | 111.050 | 125.000 | +13.950 |

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
- Evidence recall: 0.635
- Grounded answer rate: 0.909
- LLM calls/query: 2.03

## Abstention Performance
- VECTOR_RAG: correct abstention 0.000; false answer 1.000
- GRAPH_RAG: correct abstention 0.000; false answer 0.000
- HYBRID_RAG: correct abstention 0.000; false answer 0.000
- AGENTIC_HYBRID_RAG: correct abstention 1.000; false answer 0.000

## Citation Performance
- VECTOR_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000
- GRAPH_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000
- HYBRID_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000
- AGENTIC_HYBRID_RAG: citation validity 1.000; trusted citation rate 1.000; provenance completeness 1.000

## Confidence Analysis
- VECTOR_RAG: {'high': 30, 'medium': 0, 'low': 0, 'insufficient_evidence': 0}; high-confidence error rate 0.933
- GRAPH_RAG: {'high': 15, 'medium': 0, 'low': 0, 'insufficient_evidence': 15}; high-confidence error rate 0.467
- HYBRID_RAG: {'high': 17, 'medium': 0, 'low': 0, 'insufficient_evidence': 13}; high-confidence error rate 0.529
- AGENTIC_HYBRID_RAG: {'high': 15, 'medium': 5, 'low': 0, 'insufficient_evidence': 10}; high-confidence error rate 0.467

## Refinement Analysis
- VECTOR_RAG: refinement rate 0.000; success rate 0.000
- GRAPH_RAG: refinement rate 0.000; success rate 0.000
- HYBRID_RAG: refinement rate 0.000; success rate 0.000
- AGENTIC_HYBRID_RAG: refinement rate 0.100; success rate 0.667

## Latency and Cost
- VECTOR_RAG: mean 15.0 ms / median 15.0 ms / p95 32.0 ms; LLM calls/query 1.00
- GRAPH_RAG: mean 103.3 ms / median 32.0 ms / p95 309.8 ms; LLM calls/query 0.50
- HYBRID_RAG: mean 40.7 ms / median 46.0 ms / p95 62.9 ms; LLM calls/query 0.57
- AGENTIC_HYBRID_RAG: mean 59.9 ms / median 62.0 ms / p95 125.0 ms; LLM calls/query 2.03

## Agentic Wins
- `ambig-dataset`
- `ambig-method`
- `ambig-paper-a`
- `ambig-task`
- `mixed-paper-c`
- `mixed-refined-a`
- `sem-refined-a`
- `struct-datasets-a`
- `struct-datasets-b`
- `unans-missing-dataset`
- `unans-missing-limitation-b`
- `unans-missing-property`
- `unans-nonexistent-relation`

## Agentic Losses
- `mixed-approach-datasets-a`
- `mixed-attack-methods-datasets`
- `multi-hop-citation-neighborhood`
- `multi-hop-citing-datasets-a`
- `multi-hop-methods-alt-dataset`
- `multi-hop-methods-dataset`
- `sem-approach-c`
- `sem-limit-a`
- `sem-method-a`
- `sem-problem-b`
- `shared-dataset-a`
- `shared-dataset-b`
- `shared-method-a`
- `shared-task-a`
- `struct-methods-c`
- `struct-tasks-a`

## Failure Analysis
- VECTOR_RAG: {'none': 2, 'retrieval_miss': 28}
- GRAPH_RAG: {'graph_request_missing': 7, 'none': 8, 'answer_incorrect': 7, 'retrieval_miss': 2, 'execution_error': 6}
- HYBRID_RAG: {'graph_request_missing': 7, 'none': 8, 'answer_incorrect': 7, 'retrieval_miss': 2, 'execution_error': 6}
- AGENTIC_HYBRID_RAG: {'none': 10, 'retrieval_miss': 3, 'answer_incorrect': 7, 'insufficient_retrieval_evidence': 4, 'entity_not_found': 2, 'entity_ambiguous': 4}

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
- Generated at: `2026-08-19T16:16:44.235675+00:00`
