"""JSON and Markdown report writers for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.end_to_end_models import EndToEndEvaluationReport, EndToEndSystem
from evaluation.models import RetrievalEvaluationReport


def write_reports(report: RetrievalEvaluationReport, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "retrieval_report.json"
    markdown_path = output / "retrieval_report.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def write_end_to_end_reports(report: EndToEndEvaluationReport, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "end_to_end_report.json"
    markdown_path = output / "end_to_end_report.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_end_to_end_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(report: RetrievalEvaluationReport) -> str:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        "## Executive Summary",
        f"Benchmark `{report.benchmark_version}` evaluated VECTOR, GRAPH, and HYBRID retrieval with deterministic ground truth. LLM calls: {report.metadata.get('llm_calls', 0)}.",
        "",
        "## Benchmark Dataset",
        f"Cases: {len(report.case_results)}",
        "",
        "## Overall Results",
        _strategy_table(report.overall),
        "",
    ]
    for heading, key in [
        ("Semantic Results", "semantic"),
        ("Structural Results", "structural"),
        ("Shared-Entity Results", "shared_entity"),
        ("Multi-Hop Results", "multi_hop"),
        ("Mixed Results", "mixed"),
    ]:
        lines.extend([f"## {heading}", _strategy_table(report.by_category.get(key, {})), ""])

    lines.extend(
        [
            "## Hybrid Wins",
            _case_list(report.hybrid_wins),
            "",
            "## Vector Wins",
            _case_list(report.vector_wins),
            "",
            "## Graph Wins",
            _case_list(report.graph_wins),
            "",
            "## Failure Analysis",
            _failure_lines(report),
            "",
            "## Latency",
            _latency_lines(report.overall),
            "",
            "## Limitations",
            "Controlled V1 benchmarks prove mechanics and comparison logic; they do not replace a larger real-corpus evaluation.",
            "",
            "## Reproducibility",
            f"- Benchmark checksum: `{report.benchmark_checksum}`",
            f"- Run fingerprint: `{report.run_fingerprint}`",
            f"- RRF K: `{report.metadata.get('rrf_k')}`",
            f"- K values: `{report.k_values}`",
            f"- Generated at: `{report.generated_at}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_end_to_end_markdown_report(report: EndToEndEvaluationReport) -> str:
    lines = [
        "# End-to-End Graph-RAG Evaluation Report",
        "",
        "## Executive Summary",
        f"Benchmark `{report.benchmark_version}` compared VECTOR_RAG, GRAPH_RAG, HYBRID_RAG, and AGENTIC_HYBRID_RAG on this controlled benchmark. Results are not claims of statistical significance.",
        "",
        "## Benchmark",
        f"- Cases: {len({result.case_id for result in report.case_results})}",
        f"- Checksum: `{report.benchmark_checksum}`",
        "",
        "## System Variants",
        "- VECTOR_RAG: vector retrieval, answer generation, citation validation, grounding.",
        "- GRAPH_RAG: benchmark-provided graph retrieval, answer generation, citation validation, grounding.",
        "- HYBRID_RAG: benchmark-provided vector + graph retrieval with RRF, answer generation, citation validation, grounding.",
        "- AGENTIC_HYBRID_RAG: production Prompt 19 workflow with planning, sufficiency, optional refinement, answer, citations, grounding.",
        "",
        "## Overall Results",
        _end_to_end_portfolio_table(report),
        "",
        "## Prompt 20.1 Delta",
        _prompt20_1_delta_lines(report),
        "",
        "## Results by Category",
        _end_to_end_category_table(report),
        "",
        "## Vector RAG",
        _single_system_lines(report, EndToEndSystem.VECTOR_RAG),
        "",
        "## Graph RAG",
        _single_system_lines(report, EndToEndSystem.GRAPH_RAG),
        "",
        "## Hybrid RAG",
        _single_system_lines(report, EndToEndSystem.HYBRID_RAG),
        "",
        "## Agentic Hybrid Graph-RAG",
        _single_system_lines(report, EndToEndSystem.AGENTIC_HYBRID_RAG),
        "",
        "## Abstention Performance",
        _abstention_lines(report),
        "",
        "## Citation Performance",
        _citation_lines(report),
        "",
        "## Confidence Analysis",
        _confidence_lines(report),
        "",
        "## Refinement Analysis",
        _refinement_lines(report),
        "",
        "## Latency and Cost",
        _end_to_end_latency_lines(report),
        "",
        "## Agentic Wins",
        _case_list(report.agentic_wins),
        "",
        "## Agentic Losses",
        _case_list(report.agentic_losses),
        "",
        "## Failure Analysis",
        _end_to_end_failure_lines(report),
        "",
        "## Limitations",
        "Controlled benchmark results validate system mechanics and comparison logic. Fake providers do not measure live model quality, real-corpus coverage, extraction quality, or semantic claim entailment.",
        "",
        "## Reproducibility",
        f"- Evaluation version: `{report.metadata.get('evaluation_version')}`",
        f"- Run fingerprint: `{report.run_fingerprint}`",
        f"- Embedding: `{report.metadata.get('embedding_provider')}` / `{report.metadata.get('embedding_model')}`",
        f"- RRF K: `{report.metadata.get('rrf_k')}`",
        f"- Planner version: `{report.metadata.get('planner_version')}`",
        f"- Critic rules: `{report.metadata.get('critic_rules_version')}`",
        f"- Answer prompt: `{report.metadata.get('answer_generation_prompt_version')}`",
        f"- Citation validator: `{report.metadata.get('citation_validator_version')}`",
        f"- Grounding rules: `{report.metadata.get('grounding_rules_version')}`",
        f"- Generated at: `{report.generated_at}`",
        "",
    ]
    return "\n".join(lines)


def _strategy_table(strategy_metrics: dict) -> str:
    if not strategy_metrics:
        return "NOT INCLUDED"
    rows = ["| Strategy | Applicable | Hit@5 | Recall@5 | MRR |", "| --- | ---: | ---: | ---: | ---: |"]
    for strategy in ["vector", "graph", "hybrid"]:
        metrics = strategy_metrics.get(strategy)
        if metrics is None or metrics.applicable_cases == 0:
            rows.append(f"| {strategy.upper()} | 0 | 0.000 | 0.000 | 0.000 |")
            continue
        rows.append(
            f"| {strategy.upper()} | {metrics.applicable_cases} | "
            f"{metrics.hit_at_k.get(5, 0.0):.3f} | {metrics.recall_at_k.get(5, 0.0):.3f} | {metrics.mrr:.3f} |"
        )
    return "\n".join(rows)


def _case_list(case_ids: list[str]) -> str:
    return "None." if not case_ids else "\n".join(f"- `{case_id}`" for case_id in case_ids)


def _failure_lines(report: RetrievalEvaluationReport) -> str:
    failures: list[str] = []
    for case in report.case_results:
        for result in case.strategy_results.values():
            if result.failure_reason:
                failures.append(f"- `{case.case_id}` {result.strategy.value}: {result.failure_reason.value}")
    return "None." if not failures else "\n".join(failures)


def _latency_lines(strategy_metrics: dict) -> str:
    rows = []
    for strategy in ["vector", "graph", "hybrid"]:
        metrics = strategy_metrics.get(strategy)
        if metrics is None:
            rows.append(f"- {strategy.upper()}: mean 0.0 ms / median 0.0 ms / p95 0.0 ms")
        else:
            rows.append(
                f"- {strategy.upper()}: mean {metrics.mean_latency_ms:.1f} ms / "
                f"median {metrics.median_latency_ms:.1f} ms / p95 {metrics.p95_latency_ms:.1f} ms"
            )
    return "\n".join(rows)


def _end_to_end_portfolio_table(report: EndToEndEvaluationReport) -> str:
    rows = [
        "| System | Answer Accuracy | Correct Abstention | Citation Validity | High-Confidence Error | p95 Latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in EndToEndSystem:
        metrics = report.overall[system.value]
        rows.append(
            f"| {system.value} | {metrics.answer_accuracy:.3f} | "
            f"{metrics.correct_abstention_rate:.3f} | {metrics.citation_validity_rate:.3f} | "
            f"{metrics.high_confidence_error_rate:.3f} | {metrics.p95_latency_ms:.1f} ms |"
        )
    return "\n".join(rows)


def _prompt20_1_delta_lines(report: EndToEndEvaluationReport) -> str:
    payload = report.metadata.get("prompt20_baseline_metrics")
    if not isinstance(payload, dict):
        return "NOT INCLUDED"
    delta = payload.get("delta", {})
    rows = ["| Metric | Prompt 20 | Prompt 20.1 | Delta |", "| --- | ---: | ---: | ---: |"]
    for metric in [
        "answer_accuracy",
        "correct_abstention_rate",
        "false_answer_rate",
        "evidence_recall",
        "refinement_rate",
        "refinement_success_rate",
        "HIGH_count",
        "high_confidence_error_rate",
        "p95_latency_ms",
    ]:
        values = delta.get(metric, {})
        rows.append(
            f"| {metric} | {_fmt_delta_value(values.get('prompt20'))} | "
            f"{_fmt_delta_value(values.get('prompt20_1'))} | {_fmt_delta_value(values.get('delta'), signed=True)} |"
        )
    return "\n".join(rows)


def _fmt_delta_value(value, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    if isinstance(value, float):
        return f"{value:+.3f}" if signed else f"{value:.3f}"
    return str(value)


def _end_to_end_category_table(report: EndToEndEvaluationReport) -> str:
    rows = ["| Category | Vector | Graph | Hybrid | Agentic |", "| --- | ---: | ---: | ---: | ---: |"]
    for category, system_metrics in report.by_category.items():
        rows.append(
            f"| {category} | "
            f"{system_metrics['VECTOR_RAG'].answer_accuracy:.3f} | "
            f"{system_metrics['GRAPH_RAG'].answer_accuracy:.3f} | "
            f"{system_metrics['HYBRID_RAG'].answer_accuracy:.3f} | "
            f"{system_metrics['AGENTIC_HYBRID_RAG'].answer_accuracy:.3f} |"
        )
    return "\n".join(rows)


def _single_system_lines(report: EndToEndEvaluationReport, system: EndToEndSystem) -> str:
    metrics = report.overall[system.value]
    return "\n".join(
        [
            f"- Cases: {metrics.cases}",
            f"- Answer accuracy: {metrics.answer_accuracy:.3f}",
            f"- Evidence recall: {metrics.evidence_recall:.3f}",
            f"- Grounded answer rate: {metrics.grounded_answer_rate:.3f}",
            f"- LLM calls/query: {metrics.llm_calls_per_query:.2f}",
        ]
    )


def _abstention_lines(report: EndToEndEvaluationReport) -> str:
    return "\n".join(
        f"- {system.value}: correct abstention {report.overall[system.value].correct_abstention_rate:.3f}; false answer {report.overall[system.value].false_answer_rate:.3f}"
        for system in EndToEndSystem
    )


def _citation_lines(report: EndToEndEvaluationReport) -> str:
    return "\n".join(
        f"- {system.value}: citation validity {report.overall[system.value].citation_validity_rate:.3f}; trusted citation rate {report.overall[system.value].trusted_citation_rate:.3f}; provenance completeness {report.overall[system.value].provenance_completeness:.3f}"
        for system in EndToEndSystem
    )


def _confidence_lines(report: EndToEndEvaluationReport) -> str:
    rows = []
    for system in EndToEndSystem:
        metrics = report.overall[system.value]
        rows.append(
            f"- {system.value}: {metrics.confidence_distribution}; high-confidence error rate {metrics.high_confidence_error_rate:.3f}"
        )
    return "\n".join(rows)


def _refinement_lines(report: EndToEndEvaluationReport) -> str:
    rows = []
    for system in EndToEndSystem:
        metrics = report.overall[system.value]
        rows.append(
            f"- {system.value}: refinement rate {metrics.refinement_rate:.3f}; success rate {metrics.refinement_success_rate:.3f}"
        )
    return "\n".join(rows)


def _end_to_end_latency_lines(report: EndToEndEvaluationReport) -> str:
    rows = []
    for system in EndToEndSystem:
        metrics = report.overall[system.value]
        rows.append(
            f"- {system.value}: mean {metrics.mean_latency_ms:.1f} ms / median {metrics.median_latency_ms:.1f} ms / p95 {metrics.p95_latency_ms:.1f} ms; LLM calls/query {metrics.llm_calls_per_query:.2f}"
        )
    return "\n".join(rows)


def _end_to_end_failure_lines(report: EndToEndEvaluationReport) -> str:
    rows = []
    for system, failures in report.failure_analysis.items():
        rows.append(f"- {system}: {failures}")
    return "\n".join(rows) if rows else "None."
