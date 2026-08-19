import { Database, FileText, GitFork, Info } from "lucide-react";
import type { EvidenceItem, EvidencePool } from "../types/query";
import { compactId, evidenceLabel, metadataText, pageRange, safeSnippet, titleCase } from "../features/format";

interface EvidenceTabProps {
  evidence: EvidenceItem[];
  evidencePool?: EvidencePool | null;
}

export function EvidenceTab({ evidence, evidencePool }: EvidenceTabProps) {
  if (evidence.length === 0) {
    return <p className="muted">No evidence items were returned.</p>;
  }

  const poolLabels = new Map((evidencePool?.items || []).map((item) => [item.evidence.evidence_id, item.pool_id]));

  return (
    <div className="evidence-list">
      {evidence.map((item, index) => (
        <article className="evidence-item" key={item.evidence_id}>
          <div className="evidence-heading">
            <span className="evidence-label">{evidenceLabel(item, index, poolLabels.get(item.evidence_id))}</span>
            <strong>{titleCase(item.evidence_type)}</strong>
            <span>{item.source_store || item.source || "source unknown"}</span>
          </div>
          {item.evidence_type === "text" && (
            <p className="snippet">
              <FileText size={16} aria-hidden="true" />
              {safeSnippet(item.text)}
            </p>
          )}
          {(item.evidence_type === "graph_relationship" || item.evidence_type === "graph_path") && (
            <p className="snippet">
              <GitFork size={16} aria-hidden="true" />
              {safeSnippet(item.text)}
            </p>
          )}
          {item.evidence_type === "metadata" && (
            <p className="snippet">
              <Database size={16} aria-hidden="true" />
              {metadataText(item.metadata)}
            </p>
          )}
          <dl className="detail-grid compact">
            <div>
              <dt>Paper</dt>
              <dd>{item.paper_id || "Not available"}</dd>
            </div>
            <div>
              <dt>Section</dt>
              <dd>{titleCase(item.section_type)}</dd>
            </div>
            <div>
              <dt>Pages</dt>
              <dd>{pageRange(item)}</dd>
            </div>
            <div>
              <dt>Provenance</dt>
              <dd>{item.provenance?.provenance_complete === false ? "Incomplete" : "Complete"}</dd>
            </div>
          </dl>
          <details className="technical-details">
            <summary>
              <Info size={15} aria-hidden="true" />
              Technical details
            </summary>
            <pre>{metadataText({ evidence_id: compactId(item.evidence_id), entity_ids: item.entity_ids, relationship_ids: item.relationship_ids, metadata: item.metadata })}</pre>
          </details>
        </article>
      ))}
    </div>
  );
}
