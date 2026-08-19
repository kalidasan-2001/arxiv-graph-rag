import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { RetrievalWorkflowResult } from "./types/query";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ArXivGraph research UI", () => {
  it("submits a query and renders answer, confidence, citations, strategy, evidence, graph, trace, and refinement", async () => {
    let resolveFetch!: (value: unknown) => void;
    fetchMock.mockReturnValue(new Promise((resolve) => {
      resolveFetch = resolve;
    }));
    render(<App />);

    await userEvent.type(screen.getByLabelText(/ask a research question/i), "Which datasets does Paper A evaluate on?");
    await userEvent.click(screen.getByRole("button", { name: /ask/i }));

    expect(screen.getByText(/running research workflow/i)).toBeInTheDocument();
    resolveFetch(jsonResponse(successResult()));
    await screen.findByText(/Controlled Paper A evaluates on EvalSet/i);

    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/query/answer", expect.objectContaining({ method: "POST" }));
    expect(screen.getAllByLabelText(/confidence high/i).length).toBeGreaterThan(0);
    expect(screen.getByRole("tab", { name: /^Graph$/i })).toBeInTheDocument();
    expect(screen.getAllByText(/paper datasets/i).length).toBeGreaterThan(0);

    await userEvent.click(screen.getByRole("tab", { name: /sources/i }));
    expect(screen.getByText(/Citation \[1\]/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /evidence/i }));
    expect(screen.getByText(/Paper 'Controlled Paper A' evaluated on dataset 'EvalSet'/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /graph/i }));
    expect(screen.getByLabelText(/current query graph/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Controlled Paper A/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/EvalSet/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Evaluated On/i).length).toBeGreaterThan(0);

    await userEvent.click(screen.getByRole("tab", { name: /trace/i }));
    expect(screen.getByText(/Analyze Query/i)).toBeInTheDocument();
    expect(screen.getByText(/Finalize Answer/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /refinement/i }));
    expect(screen.getByText(/Retrieval refined/i)).toBeInTheDocument();
    expect(screen.getByText(/Round 2/i)).toBeInTheDocument();
  });

  it("renders abstention without manufacturing a fallback answer", async () => {
    fetchMock.mockResolvedValue(jsonResponse(abstentionResult()));
    render(<App />);

    await userEvent.type(screen.getByLabelText(/ask a research question/i), "Which papers evaluate on MissingSet?");
    await userEvent.click(screen.getByRole("button", { name: /ask/i }));

    await screen.findByText(/The requested entity could not be resolved/i);
    expect(screen.getAllByLabelText(/confidence insufficient evidence/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/fallback/i)).not.toBeInTheDocument();
  });

  it("renders disambiguation candidates read-only", async () => {
    fetchMock.mockResolvedValue(jsonResponse(disambiguationResult()));
    render(<App />);

    await userEvent.type(screen.getByLabelText(/ask a research question/i), "Which datasets does Ambiguous Paper evaluate on?");
    await userEvent.click(screen.getByRole("button", { name: /ask/i }));

    await screen.findByText(/Clarify the entity/i);
    expect(screen.getByText(/Ambiguous Paper A/i)).toBeInTheDocument();
    expect(screen.getByText(/Ambiguous Paper B/i)).toBeInTheDocument();
  });

  it("shows backend errors with a retry action", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Neo4j is temporarily unavailable." }, 503));
    fetchMock.mockResolvedValueOnce(jsonResponse(successResult()));
    render(<App />);

    await userEvent.type(screen.getByLabelText(/ask a research question/i), "Which datasets does Paper A evaluate on?");
    await userEvent.click(screen.getByRole("button", { name: /ask/i }));

    await screen.findByText(/Backend returned 503: Neo4j is temporarily unavailable/i);
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));

    await screen.findByText(/Controlled Paper A evaluates on EvalSet/i);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("allows clicking an example query into the composer", async () => {
    render(<App />);
    const examples = screen.getByLabelText(/example queries/i);

    await userEvent.click(within(examples).getByRole("button", { name: /Structural/i }));

    await waitFor(() => {
      expect(screen.getByLabelText(/ask a research question/i)).toHaveValue("Which datasets does GraphSteal evaluate on?");
    });
  });
});

function jsonResponse(payload: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: () => Promise.resolve(JSON.stringify(payload))
  };
}

function successResult(): RetrievalWorkflowResult {
  return {
    query: "Which datasets does Paper A evaluate on?",
    status: "SUCCESS",
    analysis: { intent: "paper_datasets" },
    retrieval_plan: { strategy: "graph", graph_operation: "paper_datasets", graph_limit: 5, graph_depth: 1 },
    evidence: [
      {
        evidence_id: "evidence:graph:1",
        evidence_type: "graph_relationship",
        paper_id: "paper:arxiv:a",
        source: "neo4j",
        source_store: "neo4j",
        text: "Paper 'Controlled Paper A' evaluated on dataset 'EvalSet'.",
        relationship_ids: ["rel:a-dataset"],
        entity_ids: ["paper:arxiv:a", "entity:dataset:evalset"],
        provenance: { provenance_type: "chunk", source_store: "neo4j", provenance_complete: true },
        metadata: {
          nodes: [
            { entity_id: "paper:arxiv:a", entity_type: "paper", canonical_name: "Controlled Paper A", properties: {} },
            { entity_id: "entity:dataset:evalset", entity_type: "dataset", canonical_name: "EvalSet", properties: {} }
          ],
          relationships: [
            {
              relationship_id: "rel:a-dataset",
              source_entity_id: "paper:arxiv:a",
              target_entity_id: "entity:dataset:evalset",
              relationship_type: "evaluated_on",
              confidence: 1,
              provenance_type: "chunk"
            }
          ]
        }
      },
      {
        evidence_id: "evidence:path:1",
        evidence_type: "graph_path",
        paper_id: "paper:arxiv:b",
        source: "neo4j",
        source_store: "neo4j",
        text: "Graph path: paper:Controlled Paper B -> paper:Controlled Paper A -> dataset:EvalSet.",
        relationship_ids: ["rel:b-cites-a", "rel:a-dataset"],
        entity_ids: ["paper:arxiv:b", "paper:arxiv:a", "entity:dataset:evalset"],
        metadata: {
          nodes: [
            { entity_id: "paper:arxiv:b", entity_type: "paper", canonical_name: "Controlled Paper B", properties: {} },
            { entity_id: "paper:arxiv:a", entity_type: "paper", canonical_name: "Controlled Paper A", properties: {} },
            { entity_id: "entity:dataset:evalset", entity_type: "dataset", canonical_name: "EvalSet", properties: {} }
          ],
          relationships: [
            {
              relationship_id: "rel:b-cites-a",
              source_entity_id: "paper:arxiv:b",
              target_entity_id: "paper:arxiv:a",
              relationship_type: "cites"
            },
            {
              relationship_id: "rel:a-dataset-path",
              source_entity_id: "paper:arxiv:a",
              target_entity_id: "entity:dataset:evalset",
              relationship_type: "evaluated_on"
            }
          ]
        }
      }
    ],
    evidence_pool: {
      items: [
        { pool_id: "E1", evidence: { evidence_id: "evidence:graph:1", evidence_type: "graph_relationship", source: "neo4j" } },
        { pool_id: "E2", evidence: { evidence_id: "evidence:path:1", evidence_type: "graph_path", source: "neo4j" } }
      ]
    },
    evidence_assessment: { sufficient: true, coverage: "complete", structural_coverage: true },
    evidence_history: [
      { retrieval_round: 1, strategy: "graph", evidence_count: 1, new_unique_evidence_count: 1, sufficient: false },
      { retrieval_round: 2, strategy: "graph", evidence_count: 2, new_unique_evidence_count: 1, refinement_type: "graph_depth_expansion", sufficient: true }
    ],
    retrieval_round: 2,
    refinement: { refinement_type: "graph_depth_expansion", strategy: "graph", reason_code: "expand_graph_depth", retrieval_round: 2 },
    evidence_sufficient: true,
    answer: "Controlled Paper A evaluates on EvalSet [1].",
    citations: [
      {
        citation_id: "C1",
        evidence_id: "evidence:graph:1",
        paper_id: "paper:arxiv:a",
        citation_number: 1,
        evidence_label: "E1",
        evidence_type: "graph_relationship",
        source_store: "neo4j",
        provenance_complete: true
      }
    ],
    citation_validation: { validation_status: "valid", valid_markers: ["[E1]"], invalid_markers: [] },
    grounding: { allow_answer: true, confidence: "high", reason_codes: ["strong_grounded_support"] },
    final_answer: {
      query: "Which datasets does Paper A evaluate on?",
      status: "answered",
      answer: "Controlled Paper A evaluates on EvalSet [1].",
      confidence: "high",
      citations: [
        {
          citation_id: "C1",
          evidence_id: "evidence:graph:1",
          paper_id: "paper:arxiv:a",
          citation_number: 1,
          evidence_label: "E1",
          evidence_type: "graph_relationship",
          source_store: "neo4j",
          provenance_complete: true
        }
      ],
      retrieval_rounds: 2
    },
    final_status: "answered",
    confidence: "high",
    trace: [
      { node: "analyze_query", status: "success", duration_ms: 4, metadata: { intent: "paper_datasets" } },
      { node: "execute_retrieval", status: "graph", duration_ms: 12, metadata: { strategy: "graph" } },
      { node: "finalize_answer", status: "answered", duration_ms: 1, metadata: { confidence: "high" } }
    ],
    timings: { planning: 4, retrieval: 12, finalization: 1 },
    warnings: []
  };
}

function abstentionResult(): RetrievalWorkflowResult {
  return {
    query: "Which papers evaluate on MissingSet?",
    status: "ENTITY_NOT_FOUND",
    retrieval_round: 0,
    evidence: [],
    final_status: "abstained",
    confidence: "insufficient_evidence",
    grounding: { allow_answer: false, confidence: "insufficient_evidence", reason_codes: ["entity_not_found"] },
    final_answer: {
      query: "Which papers evaluate on MissingSet?",
      status: "abstained",
      answer: "The requested entity could not be resolved in the indexed research corpus.",
      confidence: "insufficient_evidence",
      citations: [],
      retrieval_rounds: 0
    },
    trace: [{ node: "resolve_entities", status: "entity_not_found", duration_ms: 2, metadata: {} }]
  };
}

function disambiguationResult(): RetrievalWorkflowResult {
  return {
    query: "Which datasets does Ambiguous Paper evaluate on?",
    status: "REQUIRES_DISAMBIGUATION",
    final_status: "requires_disambiguation",
    confidence: "insufficient_evidence",
    evidence: [],
    ambiguous_entities: [
      {
        text: "Ambiguous Paper",
        entity_type: "paper",
        candidates: [
          { entity_id: "paper:amb:a", entity_type: "paper", canonical_name: "Ambiguous Paper A" },
          { entity_id: "paper:amb:b", entity_type: "paper", canonical_name: "Ambiguous Paper B" }
        ]
      }
    ],
    final_answer: {
      query: "Which datasets does Ambiguous Paper evaluate on?",
      status: "requires_disambiguation",
      answer: "The requested entity is ambiguous. Please select one of the available candidates.",
      confidence: "insufficient_evidence",
      citations: []
    },
    grounding: { allow_answer: false, confidence: "insufficient_evidence", reason_codes: ["entity_ambiguous"] },
    trace: [{ node: "resolve_entities", status: "requires_disambiguation", duration_ms: 2, metadata: {} }]
  };
}
