import { Database, FileText, GitGraph, ListTree, RefreshCw, Settings2 } from "lucide-react";
import { useState } from "react";
import { DiagnosticsTab } from "./DiagnosticsTab";
import { DisambiguationPanel } from "./DisambiguationPanel";
import { EvidenceTab } from "./EvidenceTab";
import { GraphTab } from "./GraphTab";
import { RefinementPanel } from "./RefinementPanel";
import { SourcesTab } from "./SourcesTab";
import { TraceTab } from "./TraceTab";
import type { RetrievalWorkflowResult } from "../types/query";

interface ResultTabsProps {
  result: RetrievalWorkflowResult | null;
}

const tabs = [
  { id: "sources", label: "Sources", icon: FileText },
  { id: "evidence", label: "Evidence", icon: Database },
  { id: "graph", label: "Graph", icon: GitGraph },
  { id: "trace", label: "Trace", icon: ListTree },
  { id: "refinement", label: "Refinement", icon: RefreshCw },
  { id: "diagnostics", label: "Diagnostics", icon: Settings2 }
] as const;

type TabId = (typeof tabs)[number]["id"];

export function ResultTabs({ result }: ResultTabsProps) {
  const [active, setActive] = useState<TabId>("sources");

  if (!result) {
    return (
      <section className="tabs-shell empty-tabs">
        <h2>Example capabilities</h2>
        <div className="capability-grid">
          <span>Semantic search</span>
          <span>Structural graph queries</span>
          <span>Multi-hop relationships</span>
          <span>Grounded citations</span>
          <span>Safe abstention</span>
        </div>
      </section>
    );
  }

  const citations = result.final_answer?.citations || result.citations || [];
  const evidence = result.evidence || [];

  return (
    <section className="tabs-shell">
      <DisambiguationPanel ambiguousEntities={result.ambiguous_entities} />
      <div className="tab-list" role="tablist" aria-label="Result details">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active === tab.id}
              className={active === tab.id ? "active" : ""}
              onClick={() => setActive(tab.id)}
            >
              <Icon size={16} aria-hidden="true" />
              {tab.label}
            </button>
          );
        })}
      </div>
      <div className="tab-panel" role="tabpanel">
        {active === "sources" && <SourcesTab citations={citations} />}
        {active === "evidence" && <EvidenceTab evidence={evidence} evidencePool={result.evidence_pool} />}
        {active === "graph" && <GraphTab evidence={evidence} />}
        {active === "trace" && <TraceTab result={result} />}
        {active === "refinement" && <RefinementPanel result={result} />}
        {active === "diagnostics" && <DiagnosticsTab result={result} />}
      </div>
    </section>
  );
}
