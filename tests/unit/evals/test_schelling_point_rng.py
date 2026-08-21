import random
from pathlib import Path
from typing import Any


def _make_eval(monkeypatch: Any, seed: int):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.api import DummyCompletionFn
    from evals.elsuite.schelling_point.eval import SchellingPoint

    return SchellingPoint(
        completion_fns=[DummyCompletionFn()],
        seed=seed,
        eval_registry_path=Path("."),
    )


def test_schelling_point_does_not_reseed_global_rng(monkeypatch: Any) -> None:
    random.seed(8675309)
    global_state = random.getstate()

    _make_eval(monkeypatch, seed=42)

    assert random.getstate() == global_state


def test_schelling_point_uses_seeded_local_rng(monkeypatch: Any) -> None:
    first = _make_eval(monkeypatch, seed=123)
    second = _make_eval(monkeypatch, seed=123)

    assert [first.rng.random() for _ in range(3)] == [second.rng.random() for _ in range(3)]
