import { postJson } from "./client";
import type { RetrievalWorkflowResult } from "../types/query";

export function answerQuery(query: string): Promise<RetrievalWorkflowResult> {
  return postJson<RetrievalWorkflowResult>("/api/v1/query/answer", { query });
}
