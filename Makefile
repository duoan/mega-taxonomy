.DEFAULT_GOAL := default

.PHONY: default install format format-cpp lint typecheck test check upgrade build clean

default: install check

install:
	uv sync --dev

lint:
	uv run python devtools/lint.py

format:
	uv run ruff format mega_taxonomy tests devtools
	$(MAKE) format-cpp

format-cpp:
	@command -v clang-format >/dev/null 2>&1 || (echo "clang-format not found; install it first" && exit 1)
	@FILES="$$(rg --files -g '*.{h,hpp,c,cc,cpp,cxx,cu,mm,metal}' include src tests bench mega_taxonomy/indexflat)"; \
	if [ -n "$$FILES" ]; then \
		echo "$$FILES" | xargs clang-format -i; \
	fi

typecheck:
	uv run basedpyright mega_taxonomy tests devtools

test:
	uv run pytest

check: lint test

upgrade:
	uv sync --upgrade --dev

build:
	uv build

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf build/
	-rm -rf build-cpp/
	-rm -rf build-cpp-omp/
	-rm -rf .ruff_cache/
	-find . -type d -name "__pycache__" -exec rm -rf {} +
