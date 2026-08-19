"""Run deterministic query-planning evaluation and write a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.domain.enums import EntityType
from app.graph.models import GraphNodeRecord
from app.retrieval.planning import (
    PlanningStatus,
    QueryAnalysisService,
    QueryPlanningService,
    RetrievalPlanner,
)


class FixtureGraphRepository:
    def __init__(self) -> None:
        self.graphsteal = GraphNodeRecord(
            entity_id="paper:arxiv:graphsteal",
            entity_type="paper",
            canonical_name="GraphSteal",
        )

    def get_entity(self, entity_id: str):
        return self.graphsteal if entity_id == self.graphsteal.entity_id else None

    def find_entities_by_canonical_name(self, canonical_name: str, *, entity_type=None, limit=20):
        if canonical_name == "GraphSteal" and entity_type == EntityType.PAPER.value:
            return [self.graphsteal]
        return []


class FixturePlannerLLM:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.last_usage = None

    @property
    def provider_name(self) -> str:
        return "fixture"

    @property
    def model_name(self) -> str:
        return "fixture-structured-analysis"

    @property
    def provider_version(self) -> str:
        return "v1"

    @property
    def temperature(self) -> float:
        return 0.0

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model):
        return response_model.model_validate(self.response)


def _planned_case(case: dict[str, Any], settings: Settings) -> dict[str, Any]:
    response = {"query": case["query"], **case["analysis"]}
    service = QueryPlanningService(
        QueryAnalysisService(FixturePlannerLLM(response), settings=settings),
        RetrievalPlanner(FixtureGraphRepository(), settings=settings),
        settings=settings,
    )
    result = service.plan(case["query"])
    plan = result.plan
    resolved_type = result.resolved_entities[0].entity_type.value if result.resolved_entities else None
    return {
        "id": case["id"],
        "category": case["category"],
        "status": result.status.value,
        "intent": result.analysis.intent.value if result.analysis else None,
        "strategy": plan.strategy.value if plan else None,
        "graph_operation": plan.graph_operation if plan else None,
        "entity_type": resolved_type,
        "intent_correct": result.analysis is not None
        and result.analysis.intent.value == case["analysis"]["intent"],
        "strategy_correct": plan is not None and plan.strategy.value == case["expected_strategy"],
        "graph_operation_correct": (plan.graph_operation if plan else None)
        == case["expected_graph_operation"],
        "entity_type_correct": resolved_type == case["expected_entity_type"],
    }


def _accuracy(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic query planning.")
    parser.add_argument("--benchmark", default="evaluation/planning_benchmark.json")
    parser.add_argument("--output", default="evaluation/results/planning_report.json")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    settings = Settings(_env_file=None)
    rows = [_planned_case(case, settings) for case in benchmark["cases"]]
    report = {
        "benchmark_version": benchmark["benchmark_version"],
        "case_count": len(rows),
        "metrics": {
            "intent_accuracy": _accuracy(rows, "intent_correct"),
            "strategy_accuracy": _accuracy(rows, "strategy_correct"),
            "graph_operation_accuracy": _accuracy(rows, "graph_operation_correct"),
            "entity_type_accuracy": _accuracy(rows, "entity_type_correct"),
        },
        "cases": rows,
        "note": "Fixture-provider evaluation validates planner software behavior, not live model accuracy.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
