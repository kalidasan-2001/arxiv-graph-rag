import type { AnswerCitation, EvidenceItem } from "../types/query";

export function titleCase(value?: string | null): string {
  if (!value) {
    return "Not available";
  }
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function compactId(value?: string | null): string {
  if (!value) {
    return "Not available";
  }
  if (value.length <= 28) {
    return value;
  }
  return `${value.slice(0, 14)}...${value.slice(-8)}`;
}

export function pageRange(item: EvidenceItem | AnswerCitation): string {
  if (item.page_start && item.page_end) {
    return item.page_start === item.page_end ? `${item.page_start}` : `${item.page_start}-${item.page_end}`;
  }
  if (item.page_start) {
    return `${item.page_start}`;
  }
  return "Not available";
}

export function evidenceLabel(evidence: EvidenceItem, index: number, poolLabel?: string): string {
  return poolLabel || `E${index + 1}`;
}

export function safeSnippet(text?: string | null, maxLength = 420): string {
  if (!text) {
    return "No text snippet returned for this evidence item.";
  }
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}...` : text;
}

export function metadataText(value: unknown): string {
  if (value === null || value === undefined) {
    return "Not available";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
