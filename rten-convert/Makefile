.PHONY: check
check: checkformat lint typecheck

.PHONY: checkformat
checkformat:
	ruff format --check rten_convert

.PHONY: format
format:
	ruff format rten_convert

.PHONY: lint
lint:
	ruff check rten_convert

.PHONY: typecheck
typecheck:
	mypy rten_convert
