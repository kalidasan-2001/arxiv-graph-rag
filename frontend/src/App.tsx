import { useState } from "react";
import { answerQuery } from "./api/query";
import { AnswerPanel } from "./components/AnswerPanel";
import { ExecutionPanel } from "./components/ExecutionPanel";
import { QueryComposer } from "./components/QueryComposer";
import { ResultTabs } from "./components/ResultTabs";
import type { RetrievalWorkflowResult } from "./types/query";

export default function App() {
  const [result, setResult] = useState<RetrievalWorkflowResult | null>(null);
  const [lastQuery, setLastQuery] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runQuery(query: string) {
    setLoading(true);
    setError(null);
    setLastQuery(query);
    try {
      setResult(await answerQuery(query));
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Query failed.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setResult(null);
    setLastQuery("");
    setError(null);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ArXivGraph</p>
          <h1>Scientific Research Intelligence</h1>
        </div>
        <span className="api-pill">FastAPI · Graph-RAG</span>
      </header>

      <QueryComposer disabled={loading} onSubmit={runQuery} onReset={reset} />

      {lastQuery && (
        <section className="query-context" aria-label="Submitted query">
          <span>Question</span>
          <strong>{lastQuery}</strong>
        </section>
      )}

      <div className="workspace-grid">
        <AnswerPanel result={result} loading={loading} error={error} onRetry={() => lastQuery && runQuery(lastQuery)} />
        <ExecutionPanel result={result} />
      </div>

      <ResultTabs result={result} />
    </main>
  );
}
