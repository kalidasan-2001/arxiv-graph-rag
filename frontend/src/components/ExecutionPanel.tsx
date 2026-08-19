import type { ReactNode } from "react";
import { Activity, GitBranch, Layers3, Route, SearchCheck } from "lucide-react";
import { ConfidenceBadge } from "./ConfidenceBadge";
import type { RetrievalWorkflowResult } from "../types/query";
import { titleCase } from "../features/format";

interface ExecutionPanelProps {
  result: RetrievalWorkflowResult | null;
}

export function ExecutionPanel({ result }: ExecutionPanelProps) {
  const plan = result?.retrieval_plan;
  const analysis = result?.analysis;
  const evidenceCount = result?.evidence?.length || 0;
  const totalMs = Object.values(result?.timings || {}).reduce((sum, value) => sum + value, 0);

  return (
    <aside className="execution-panel" aria-label="How it worked">
      <p className="eyebrow">How it worked</p>
      <Metric icon={<Activity size={17} />} label="Intent" value={analysis?.intent || "Not available"} />
      <Metric icon={<SearchCheck size={17} />} label="Strategy" value={plan?.strategy || "Not available"} detail={plan ? "Automatically selected" : undefined} />
      <Metric icon={<GitBranch size={17} />} label="Graph operation" value={plan?.graph_operation || "None"} />
      <Metric icon={<Route size={17} />} label="Retrieval rounds" value={String(result?.retrieval_round ?? 0)} />
      <Metric icon={<Layers3 size={17} />} label="Evidence" value={String(evidenceCount)} />
      <div className="execution-confidence">
        <span>Confidence</span>
        <ConfidenceBadge confidence={result?.confidence || result?.final_answer?.confidence} />
      </div>
      {totalMs > 0 && <p className="timing-total">Total observed node time: {totalMs} ms</p>}
    </aside>
  );
}

function Metric({ icon, label, value, detail }: { icon: ReactNode; label: string; value: string; detail?: string }) {
  return (
    <div className="metric">
      <span className="metric-icon" aria-hidden="true">
        {icon}
      </span>
      <div>
        <span>{label}</span>
        <strong>{titleCase(value)}</strong>
        {detail && <small>{detail}</small>}
      </div>
    </div>
  );
}
