"""List every registered backend: ``python -m src.components``."""

from src.components import describe_all


def main() -> int:
    print(describe_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
