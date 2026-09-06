from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptlab.io import load_tasks, load_yaml, write_csv, write_json
from promptlab.models import Aggregate, Task, Trial


def test_task_deserialisation_and_trial_serialisation() -> None:
    task = Task.from_dict(
        {
            "id": "x",
            "kind": "classification",
            "category": "classification",
            "instruction": "Classify",
            "expected": "a",
            "labels": ["a", "b"],
        }
    )
    assert task.difficulty == 0.5
    trial = Trial(
        "x",
        "classification",
        "classification",
        "development",
        "p",
        "m",
        "v",
        0,
        "h",
        10,
        1.0,
        True,
        5.0,
        0.0,
    )
    assert trial.to_dict()["passed"] is True
    aggregate = Aggregate("m", "v", 1, 1.0, 1.0, 5.0, 0.0, 10.0)
    assert aggregate.to_dict()["variant"] == "v"


def test_loaders_validate_input(tmp_path: Path) -> None:
    line = {
        "id": "x",
        "kind": "classification",
        "category": "classification",
        "instruction": "Classify",
        "expected": "a",
    }
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert load_tasks(dataset)[0].id == "x"
    dataset.write_text(json.dumps(line) + "\n" + json.dumps(line), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_tasks(dataset)
    dataset.write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match=":1"):
        load_tasks(dataset)


def test_yaml_and_writers(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    assert load_yaml(config) == {"seed": 42}
    config.write_text("- bad\n- shape\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(config)
    json_path, csv_path, empty_path = tmp_path / "x.json", tmp_path / "x.csv", tmp_path / "e.csv"
    write_json(json_path, {"ok": True})
    write_csv(csv_path, [{"a": 1, "b": 2}])
    write_csv(empty_path, [])
    assert json.loads(json_path.read_text())["ok"] is True
    assert "a,b" in csv_path.read_text()
    assert empty_path.read_text() == ""
