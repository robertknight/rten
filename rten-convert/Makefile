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

# See https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives
.PHONY: release
release: check
	rm -rf dist/
	python -m build
	python -m twine upload dist/*
