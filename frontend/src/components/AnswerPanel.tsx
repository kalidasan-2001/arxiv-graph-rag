import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";
import { ConfidenceBadge } from "./ConfidenceBadge";
import type { RetrievalWorkflowResult } from "../types/query";
import { titleCase } from "../features/format";

interface AnswerPanelProps {
  result: RetrievalWorkflowResult | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export function AnswerPanel({ result, loading, error, onRetry }: AnswerPanelProps) {
  if (loading) {
    return (
      <section className="answer-panel" aria-live="polite">
        <p className="eyebrow">Research workflow</p>
        <h2>Running research workflow...</h2>
        <p className="muted">The backend will return the completed safe trace when the answer is ready.</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="answer-panel error-state" aria-live="assertive">
        <AlertTriangle size={22} aria-hidden="true" />
        <div>
          <p className="eyebrow">Backend error</p>
          <h2>{error}</h2>
          <button className="secondary-button" type="button" onClick={onRetry}>
            Retry
          </button>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="answer-panel empty-state">
        <p className="eyebrow">ArXivGraph</p>
        <h2>Scientific research intelligence over indexed papers.</h2>
        <p>
          Ask one question to inspect semantic retrieval, graph reasoning, multi-hop relationships, grounded citations,
          and safe abstention in a single workspace.
        </p>
      </section>
    );
  }

  const final = result.final_answer;
  const status = final?.status || result.final_status || "abstained";
  const answer = final?.answer || result.answer || result.grounding?.abstention_reason || "The backend did not return answer text.";
  const confidence = final?.confidence || result.confidence;
  const reasonCodes = result.grounding?.reason_codes || final?.grounding?.reason_codes || [];
  const citations = final?.citations || result.citations || [];
  const Icon = status === "answered" ? CheckCircle2 : status === "requires_disambiguation" ? HelpCircle : AlertTriangle;

  return (
    <section className="answer-panel" aria-live="polite">
      <div className="answer-heading">
        <div>
          <p className="eyebrow">Answer</p>
          <h2>{titleCase(status)}</h2>
        </div>
        <ConfidenceBadge confidence={confidence} />
      </div>
      <div className={`status-line status-${status}`}>
        <Icon size={18} aria-hidden="true" />
        <span>{status === "answered" ? "Grounded answer returned" : titleCase(status)}</span>
      </div>
      <p className="answer-text">{answer}</p>
      <div className="answer-meta">
        <span>{citations.length} trusted citations</span>
        <span>{result.retrieval_round || final?.retrieval_rounds || 0} retrieval rounds</span>
      </div>
      {reasonCodes.length > 0 && (
        <div className="reason-list" aria-label="Grounding reasons">
          {reasonCodes.map((reason) => (
            <span key={reason}>{titleCase(reason)}</span>
          ))}
        </div>
      )}
      {(result.warnings || []).length > 0 && (
        <div className="warning-list">
          {(result.warnings || []).map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      )}
    </section>
  );
}
