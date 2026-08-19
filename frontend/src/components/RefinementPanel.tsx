import { RefreshCw } from "lucide-react";
import type { RetrievalWorkflowResult } from "../types/query";
import { titleCase } from "../features/format";

interface RefinementPanelProps {
  result: RetrievalWorkflowResult;
}

export function RefinementPanel({ result }: RefinementPanelProps) {
  const history = result.evidence_history || [];
  if ((result.retrieval_round || 0) <= 1 && history.length <= 1) {
    return <p className="muted">No retrieval refinement was needed.</p>;
  }

  return (
    <section className="refinement-panel" aria-label="Retrieval refinement">
      <h3>
        <RefreshCw size={17} aria-hidden="true" />
        Retrieval refined
      </h3>
      <div className="round-list">
        {history.map((round) => (
          <div className="round-item" key={round.retrieval_round}>
            <span>Round {round.retrieval_round}</span>
            <strong>{titleCase(round.strategy)}</strong>
            <small>
              {round.evidence_count ?? 0} evidence items
              {round.new_unique_evidence_count !== undefined ? `, +${round.new_unique_evidence_count} new` : ""}
            </small>
            {round.refinement_type && <em>{titleCase(round.refinement_type)}</em>}
          </div>
        ))}
      </div>
      {result.refinement && (
        <dl className="detail-grid compact">
          <div>
            <dt>Type</dt>
            <dd>{titleCase(result.refinement.refinement_type)}</dd>
          </div>
          <div>
            <dt>Reason</dt>
            <dd>{titleCase(result.refinement.reason_code)}</dd>
          </div>
          <div>
            <dt>Strategy</dt>
            <dd>{titleCase(result.refinement.strategy)}</dd>
          </div>
        </dl>
      )}
    </section>
  );
}
