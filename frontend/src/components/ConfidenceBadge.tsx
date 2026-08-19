import { ShieldCheck, ShieldQuestion, ShieldX } from "lucide-react";
import type { ConfidenceLevel } from "../types/query";
import { titleCase } from "../features/format";

interface ConfidenceBadgeProps {
  confidence?: ConfidenceLevel | null;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const value = confidence || "insufficient_evidence";
  const Icon = value === "high" ? ShieldCheck : value === "insufficient_evidence" ? ShieldX : ShieldQuestion;

  return (
    <span className={`confidence confidence-${value}`} aria-label={`Confidence ${titleCase(value)}`}>
      <Icon aria-hidden="true" size={16} />
      {value.toUpperCase()}
    </span>
  );
}
