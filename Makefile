.PHONY: install check test ci-local bench clean

install:
	uv sync

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

test:
	uv run pytest -n auto -m "not slow"

ci-local:
	uv run pytest -n auto

bench:
	uv run python -m benchmarks.bench_attack

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
