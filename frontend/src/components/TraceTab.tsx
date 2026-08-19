import { CheckCircle2, Clock3, XCircle } from "lucide-react";
import type { RetrievalWorkflowResult, WorkflowTraceEvent } from "../types/query";
import { metadataText, titleCase } from "../features/format";

interface TraceTabProps {
  result: RetrievalWorkflowResult;
}

export function TraceTab({ result }: TraceTabProps) {
  const trace = result.trace || [];
  if (trace.length === 0) {
    return <p className="muted">No safe workflow trace was returned.</p>;
  }

  return (
    <div className="trace-list" aria-label="Safe workflow trace">
      {trace.map((event, index) => (
        <TraceItem key={`${event.node}-${index}`} event={event} />
      ))}
      {Object.keys(result.timings || {}).length > 0 && (
        <section className="timing-breakdown" aria-label="Timing breakdown">
          <h3>Timing</h3>
          {Object.entries(result.timings || {}).map(([name, duration]) => (
            <div key={name}>
              <span>{titleCase(name)}</span>
              <strong>{duration} ms</strong>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

function TraceItem({ event }: { event: WorkflowTraceEvent }) {
  const failed = event.status.toLowerCase().includes("failed") || event.status.toLowerCase().includes("error");
  const Icon = failed ? XCircle : CheckCircle2;
  return (
    <article className="trace-item">
      <Icon size={18} aria-hidden="true" />
      <div>
        <div className="trace-title">
          <strong>{titleCase(event.node)}</strong>
          <span>{titleCase(event.status)}</span>
          <span>
            <Clock3 size={13} aria-hidden="true" />
            {event.duration_ms} ms
          </span>
        </div>
        {event.metadata && Object.keys(event.metadata).length > 0 && (
          <details>
            <summary>Safe metadata</summary>
            <pre>{metadataText(event.metadata)}</pre>
          </details>
        )}
      </div>
    </article>
  );
}
