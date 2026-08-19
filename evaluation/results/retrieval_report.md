# Retrieval Evaluation Report

## Executive Summary
Benchmark `v1` evaluated VECTOR, GRAPH, and HYBRID retrieval with deterministic ground truth. LLM calls: 0.

## Benchmark Dataset
Cases: 10

## Overall Results
| Strategy | Applicable | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| VECTOR | 10 | 0.300 | 0.300 | 0.300 |
| GRAPH | 8 | 1.000 | 1.000 | 0.938 |
| HYBRID | 8 | 1.000 | 1.000 | 0.938 |

## Semantic Results
| Strategy | Applicable | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| VECTOR | 2 | 1.000 | 1.000 | 1.000 |
| GRAPH | 0 | 0.000 | 0.000 | 0.000 |
| HYBRID | 0 | 0.000 | 0.000 | 0.000 |

## Structural Results
| Strategy | Applicable | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| VECTOR | 3 | 0.000 | 0.000 | 0.000 |
| GRAPH | 3 | 1.000 | 1.000 | 1.000 |
| HYBRID | 3 | 1.000 | 1.000 | 1.000 |

## Shared-Entity Results
| Strategy | Applicable | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| VECTOR | 2 | 0.000 | 0.000 | 0.000 |
| GRAPH | 2 | 1.000 | 1.000 | 1.000 |
| HYBRID | 2 | 1.000 | 1.000 | 1.000 |

## Multi-Hop Results
| Strategy | Applicable | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| VECTOR | 2 | 0.000 | 0.000 | 0.000 |
| GRAPH | 2 | 1.000 | 1.000 | 1.000 |
| HYBRID | 2 | 1.000 | 1.000 | 1.000 |

## Mixed Results
| Strategy | Applicable | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| VECTOR | 1 | 1.000 | 1.000 | 1.000 |
| GRAPH | 1 | 1.000 | 1.000 | 0.500 |
| HYBRID | 1 | 1.000 | 1.000 | 0.500 |

## Hybrid Wins
- `struct-datasets-a`
- `struct-methods-a`
- `struct-tasks-a`
- `shared-dataset-a`
- `shared-method-a`
- `multi-hop-citing-datasets-a`
- `multi-hop-methods-dataset`

## Vector Wins
- `sem-method-a`
- `sem-limit-a`
- `mixed-dataset-attack`

## Graph Wins
- `struct-datasets-a`
- `struct-methods-a`
- `struct-tasks-a`
- `shared-dataset-a`
- `shared-method-a`
- `multi-hop-citing-datasets-a`
- `multi-hop-methods-dataset`

## Failure Analysis
- `sem-method-a` graph: GRAPH_NOT_APPLICABLE
- `sem-method-a` hybrid: GRAPH_NOT_APPLICABLE
- `sem-limit-a` graph: GRAPH_NOT_APPLICABLE
- `sem-limit-a` hybrid: GRAPH_NOT_APPLICABLE
- `struct-datasets-a` vector: RELEVANT_NOT_IN_TOP_K
- `struct-methods-a` vector: RELEVANT_NOT_IN_TOP_K
- `struct-tasks-a` vector: RELEVANT_NOT_IN_TOP_K
- `shared-dataset-a` vector: RELEVANT_NOT_IN_TOP_K
- `shared-method-a` vector: RELEVANT_NOT_IN_TOP_K
- `multi-hop-citing-datasets-a` vector: RELEVANT_NOT_IN_TOP_K
- `multi-hop-methods-dataset` vector: RELEVANT_NOT_IN_TOP_K

## Latency
- VECTOR: mean 12.4 ms / median 15.0 ms / p95 24.3 ms
- GRAPH: mean 84.3 ms / median 63.0 ms / p95 179.4 ms
- HYBRID: mean 37.0 ms / median 31.0 ms / p95 47.0 ms

## Limitations
Controlled V1 benchmarks prove mechanics and comparison logic; they do not replace a larger real-corpus evaluation.

## Reproducibility
- Benchmark checksum: `a50c80b2ec44be1c97e3e65f8cc039f4b66f5428a9be4b01c9676ff673aa48a3`
- Run fingerprint: `8eb4c8922d8ed5474cf517b8c77427a73d334738dac2c82826a1bc93e257c27b`
- RRF K: `60`
- K values: `[1, 3, 5, 10]`
- Generated at: `2026-08-19T13:27:03.984914+00:00`
