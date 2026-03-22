from __future__ import annotations

from demo_project.utils import add


def test_add() -> None:
    assert add(2, 3) == 5
