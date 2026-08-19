import type { RetrievalWorkflowResult } from "../types/query";
import { metadataText, titleCase } from "../features/format";

interface DiagnosticsTabProps {
  result: RetrievalWorkflowResult;
}

export function DiagnosticsTab({ result }: DiagnosticsTabProps) {
  return (
    <div className="diagnostics">
      <dl className="detail-grid">
        <div>
          <dt>Intent</dt>
          <dd>{result.analysis?.intent || "Not available"}</dd>
        </div>
        <div>
          <dt>Strategy</dt>
          <dd>{titleCase(result.retrieval_plan?.strategy)}</dd>
        </div>
        <div>
          <dt>Graph operation</dt>
          <dd>{result.retrieval_plan?.graph_operation || "None"}</dd>
        </div>
        <div>
          <dt>Citation validation</dt>
          <dd>{titleCase(result.citation_validation?.validation_status)}</dd>
        </div>
        <div>
          <dt>Evidence sufficiency</dt>
          <dd>{String(result.evidence_sufficient ?? result.evidence_assessment?.sufficient ?? "Not available")}</dd>
        </div>
        <div>
          <dt>Grounding reasons</dt>
          <dd>{(result.grounding?.reason_codes || []).map(titleCase).join(", ") || "None"}</dd>
        </div>
      </dl>
      <pre>{metadataText({ warnings: result.warnings, errors: result.errors, missing_information: result.missing_information })}</pre>
    </div>
  );
}
