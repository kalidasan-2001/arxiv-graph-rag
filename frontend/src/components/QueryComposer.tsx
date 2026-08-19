import { RotateCcw, SendHorizontal, X } from "lucide-react";
import { FormEvent, KeyboardEvent, useState } from "react";
import { exampleQueries } from "../features/examples";

interface QueryComposerProps {
  disabled: boolean;
  onSubmit: (query: string) => void;
  onReset: () => void;
}

export function QueryComposer({ disabled, onSubmit, onReset }: QueryComposerProps) {
  const [query, setQuery] = useState("");

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSubmit(trimmed);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <section className="composer">
      <div className="composer-row">
        <label id="query-label" htmlFor="research-query">
          Ask a research question
        </label>
        <button className="ghost-button" type="button" onClick={onReset} disabled={disabled} aria-label="Clear result">
          <RotateCcw size={16} aria-hidden="true" />
          Reset
        </button>
      </div>
      <form onSubmit={submit} className="query-form">
        <textarea
          id="research-query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about methods, datasets, citations, shared entities, or paper relationships..."
          disabled={disabled}
          rows={3}
        />
        <div className="composer-actions">
          <button className="icon-button" type="button" onClick={() => setQuery("")} disabled={disabled || !query} aria-label="Clear query">
            <X size={18} aria-hidden="true" />
          </button>
          <button className="primary-button" type="submit" disabled={disabled || !query.trim()}>
            <SendHorizontal size={18} aria-hidden="true" />
            {disabled ? "Running" : "Ask"}
          </button>
        </div>
      </form>
      <div className="examples" aria-label="Example queries">
        {exampleQueries.map((example) => (
          <button
            key={example.label}
            type="button"
            className="example-button"
            onClick={() => setQuery(example.query)}
            disabled={disabled}
          >
            <span>{example.label}</span>
            {example.query}
          </button>
        ))}
      </div>
    </section>
  );
}
