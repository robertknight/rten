.PHONY: check
check: format lint typecheck

.PHONY: format
format:
	ruff format rten_convert

.PHONY: lint
lint:
	ruff check rten_convert

.PHONY: typecheck
typecheck:
	mypy rten_convert
