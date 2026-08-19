import { HelpCircle } from "lucide-react";
import type { AmbiguousEntity } from "../types/query";
import { compactId, titleCase } from "../features/format";

interface DisambiguationPanelProps {
  ambiguousEntities?: AmbiguousEntity[];
}

export function DisambiguationPanel({ ambiguousEntities = [] }: DisambiguationPanelProps) {
  if (ambiguousEntities.length === 0) {
    return null;
  }

  return (
    <section className="disambiguation-panel" aria-label="Disambiguation candidates">
      <h3>
        <HelpCircle size={18} aria-hidden="true" />
        Clarify the entity
      </h3>
      <p className="muted">The backend returned candidates. V1 does not expose a follow-up selection endpoint, so refine the query with a more specific name or ID.</p>
      {ambiguousEntities.map((item, index) => (
        <div className="candidate-group" key={`${item.text}-${index}`}>
          <strong>{item.text || "Ambiguous entity"}</strong>
          <div className="candidate-list">
            {(item.candidates || []).map((candidate) => (
              <article key={candidate.entity_id || candidate.canonical_name}>
                <span>{candidate.canonical_name || "Unnamed candidate"}</span>
                <small>{titleCase(candidate.entity_type)} · {compactId(candidate.entity_id)}</small>
              </article>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
