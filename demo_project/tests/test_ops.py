from __future__ import annotations

from demo_project.ops import sum_numbers


def test_sum_numbers() -> None:
    assert sum_numbers(4, 5) == 9
