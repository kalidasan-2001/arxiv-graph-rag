import json

import pytest
from pydantic import ValidationError

from evaluation.models import RetrievalBenchmarkDataset
from evaluation.runner import validate_benchmark_file


def test_benchmark_file_validates() -> None:
    dataset = validate_benchmark_file("evaluation/retrieval_benchmark.json")

    assert dataset.benchmark_version == "v1"
    assert len(dataset.cases) >= 10


def test_duplicate_case_ids_are_rejected() -> None:
    data = {
        "cases": [
            {
                "id": "dup",
                "category": "semantic",
                "question": "one",
                "expected_targets": [{"target_type": "chunk", "target_id": "chunk:a"}],
                "relevance_target_type": "chunk",
            },
            {
                "id": "dup",
                "category": "semantic",
                "question": "two",
                "expected_targets": [{"target_type": "chunk", "target_id": "chunk:b"}],
                "relevance_target_type": "chunk",
            },
        ]
    }

    with pytest.raises(ValidationError, match="duplicate benchmark case ids"):
        RetrievalBenchmarkDataset.model_validate(data)


def test_invalid_category_target_and_empty_question_are_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "bad",
                        "category": "unknown",
                        "question": "",
                        "expected_targets": [{"target_type": "unknown", "target_id": "x"}],
                        "relevance_target_type": "chunk",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        validate_benchmark_file(path)


def test_invalid_graph_operation_is_rejected() -> None:
    data = {
        "cases": [
            {
                "id": "bad-graph",
                "category": "structural",
                "question": "Which datasets?",
                "graph_request": {"operation": "not_real", "entity_id": "paper:a"},
                "expected_targets": [{"target_type": "relationship", "target_id": "rel:a"}],
                "relevance_target_type": "relationship",
            }
        ]
    }

    with pytest.raises(ValidationError):
        RetrievalBenchmarkDataset.model_validate(data)


def test_missing_expected_targets_are_rejected() -> None:
    data = {
        "cases": [
            {
                "id": "missing",
                "category": "semantic",
                "question": "What does it say?",
                "expected_targets": [],
                "relevance_target_type": "chunk",
            }
        ]
    }

    with pytest.raises(ValidationError, match="expected_targets must not be empty"):
        RetrievalBenchmarkDataset.model_validate(data)
