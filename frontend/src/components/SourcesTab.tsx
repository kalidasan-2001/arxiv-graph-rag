import { FileText } from "lucide-react";
import { useState } from "react";
import type { AnswerCitation } from "../types/query";
import { compactId, metadataText, pageRange, titleCase } from "../features/format";

interface SourcesTabProps {
  citations: AnswerCitation[];
}

export function SourcesTab({ citations }: SourcesTabProps) {
  const [open, setOpen] = useState<string | null>(citations[0]?.citation_id || null);

  if (citations.length === 0) {
    return <p className="muted">No trusted citations were returned.</p>;
  }

  return (
    <div className="source-list">
      {citations.map((citation) => {
        const isOpen = open === citation.citation_id;
        return (
          <article className="source-item" key={citation.citation_id}>
            <button className="source-summary" type="button" onClick={() => setOpen(isOpen ? null : citation.citation_id)}>
              <FileText size={18} aria-hidden="true" />
              <span>Citation [{citation.citation_number || citation.evidence_label || "?"}]</span>
              <strong>{titleCase(citation.evidence_type || "evidence")}</strong>
            </button>
            {isOpen && (
              <dl className="detail-grid">
                <div>
                  <dt>Paper</dt>
                  <dd>{citation.paper_id || "Not available"}</dd>
                </div>
                <div>
                  <dt>Section</dt>
                  <dd>{titleCase(citation.section_type)}</dd>
                </div>
                <div>
                  <dt>Pages</dt>
                  <dd>{pageRange(citation)}</dd>
                </div>
                <div>
                  <dt>Source store</dt>
                  <dd>{citation.source_store || "Not available"}</dd>
                </div>
                <div>
                  <dt>Evidence ID</dt>
                  <dd>{compactId(citation.evidence_id)}</dd>
                </div>
                <div>
                  <dt>Provenance</dt>
                  <dd>{citation.provenance_complete === false ? "Incomplete" : "Complete"}</dd>
                </div>
                {citation.metadata && Object.keys(citation.metadata).length > 0 && (
                  <div className="wide-detail">
                    <dt>Metadata</dt>
                    <dd>
                      <pre>{metadataText(citation.metadata)}</pre>
                    </dd>
                  </div>
                )}
              </dl>
            )}
          </article>
        );
      })}
    </div>
  );
}
