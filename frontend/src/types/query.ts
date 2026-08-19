export type ConfidenceLevel = "high" | "medium" | "low" | "insufficient_evidence";
export type FinalStatus = "answered" | "abstained" | "requires_disambiguation" | "failed";
export type EvidenceType = "text" | "graph_relationship" | "graph_path" | "metadata";

export interface RetrievalWorkflowResult {
  query: string;
  status: string;
  analysis?: StructuredQueryAnalysis | null;
  planning_status?: string | null;
  resolved_entities?: ResolvedEntity[];
  ambiguous_entities?: AmbiguousEntity[];
  retrieval_plan?: RetrievalPlan | null;
  retrieval_result?: HybridRetrievalResult | null;
  evidence?: EvidenceItem[];
  evidence_pool?: EvidencePool | null;
  evidence_assessment?: EvidenceAssessment | null;
  retrieval_round?: number;
  refinement?: RetrievalRefinement | null;
  evidence_history?: EvidenceRoundSummary[];
  evidence_sufficient?: boolean | null;
  missing_information?: string[];
  refinement_reason?: string | null;
  answer?: string | null;
  citations?: AnswerCitation[];
  citation_validation?: CitationValidationResult | null;
  grounding?: GroundingDecision | null;
  final_answer?: FinalResearchAnswer | null;
  final_status?: FinalStatus | null;
  confidence?: ConfidenceLevel | null;
  warnings?: string[];
  errors?: WorkflowError[];
  trace?: WorkflowTraceEvent[];
  timings?: Record<string, number>;
}

export interface StructuredQueryAnalysis {
  query?: string;
  intent?: string;
  semantic_retrieval_required?: boolean;
  structural_retrieval_required?: boolean;
  entities?: Array<{ text: string; entity_type?: string | null }>;
  planning_confidence?: number;
  [key: string]: unknown;
}

export interface ResolvedEntity {
  text?: string;
  entity_id?: string;
  entity_type?: string;
  canonical_name?: string;
  [key: string]: unknown;
}

export interface AmbiguousEntity {
  text?: string;
  entity_type?: string;
  candidates?: Array<{
    entity_id?: string;
    entity_type?: string;
    canonical_name?: string;
    properties?: Record<string, unknown>;
  }>;
  [key: string]: unknown;
}

export interface RetrievalPlan {
  strategy?: string;
  query?: string;
  entity_ids?: string[];
  graph_operation?: string | null;
  graph_request?: Record<string, unknown> | null;
  vector_top_k?: number | null;
  graph_limit?: number | null;
  graph_depth?: number | null;
  final_top_k?: number | null;
  requested_graph_operations?: string[];
  requires_multiple_graph_operations?: boolean;
  [key: string]: unknown;
}

export interface HybridRetrievalResult {
  strategy?: string;
  diagnostics?: Record<string, unknown>;
  warnings?: string[];
}

export interface EvidenceAssessment {
  sufficient?: boolean;
  coverage?: string;
  missing_information?: string[];
  unsupported_requirements?: string[];
  recommended_refinement_type?: string | null;
  semantic_coverage?: boolean | null;
  structural_coverage?: boolean | null;
  [key: string]: unknown;
}

export interface RetrievalRefinement {
  refinement_type?: string;
  strategy?: string;
  vector_top_k?: number | null;
  graph_limit?: number | null;
  graph_depth?: number | null;
  reason_code?: string;
  retrieval_round?: number;
}

export interface EvidenceRoundSummary {
  retrieval_round: number;
  strategy?: string;
  evidence_count?: number;
  new_unique_evidence_count?: number;
  refinement_type?: string | null;
  sufficient?: boolean | null;
}

export interface EvidencePool {
  items?: Array<{ pool_id: string; evidence: EvidenceItem }>;
}

export interface EvidenceItem {
  evidence_id: string;
  evidence_type: EvidenceType | string;
  paper_id?: string | null;
  paper_version_id?: string | null;
  chunk_id?: string | null;
  section_id?: string | null;
  section_type?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  entity_ids?: string[];
  relationship_ids?: string[];
  source_chunk_ids?: string[];
  text?: string | null;
  score?: number | null;
  source?: string;
  source_store?: string | null;
  provenance?: EvidenceProvenance | null;
  supporting_text_evidence_ids?: string[];
  metadata?: Record<string, unknown>;
}

export interface EvidenceProvenance {
  provenance_type?: string;
  source_store?: string;
  paper_id?: string | null;
  chunk_ids?: string[];
  relationship_ids?: string[];
  provenance_complete?: boolean;
  warnings?: string[];
  [key: string]: unknown;
}

export interface AnswerCitation {
  citation_id: string;
  evidence_id: string;
  paper_id?: string | null;
  chunk_id?: string | null;
  label?: string;
  citation_number?: number | null;
  evidence_label?: string | null;
  evidence_type?: EvidenceType | string | null;
  section_type?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  entity_ids?: string[];
  relationship_ids?: string[];
  source_chunk_ids?: string[];
  provenance_complete?: boolean | null;
  provenance_warnings?: string[];
  source_store?: string | null;
  metadata?: Record<string, unknown>;
}

export interface CitationValidationResult {
  validation_status?: string;
  valid_markers?: string[];
  invalid_markers?: unknown[];
  warnings?: string[];
}

export interface GroundingDecision {
  allow_answer?: boolean;
  confidence?: ConfidenceLevel;
  reason_codes?: string[];
  warnings?: string[];
  abstention_reason?: string | null;
  diagnostics?: Record<string, unknown>;
}

export interface FinalResearchAnswer {
  query: string;
  status: FinalStatus;
  answer: string;
  citations?: AnswerCitation[];
  confidence: ConfidenceLevel;
  grounding?: GroundingDecision;
  missing_information?: string[];
  warnings?: string[];
  retrieval_rounds?: number;
}

export interface WorkflowTraceEvent {
  node: string;
  status: string;
  duration_ms: number;
  metadata?: Record<string, unknown>;
}

export interface WorkflowError {
  node?: string;
  error_type?: string;
  message?: string;
}
