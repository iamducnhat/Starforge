from __future__ import annotations

from ops import sum_numbers
from utils import add


def main() -> None:
    print(f"add(2, 3) = {add(2, 3)}")
    print(f"sum_numbers(2, 3) = {sum_numbers(2, 3)}")


if __name__ == "__main__":
    main()
