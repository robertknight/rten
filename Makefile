.PHONY: all
all:
	cargo build --workspace

.PHONY: cli
cli:
	cargo build -p rten-cli --release

# Build with ThinLTO disabled. This makes the build faster but the binary will
# be larger and slower. Performance is still usable though, unlike a debug
# build.
.PHONY: cli-no-lto
cli-no-lto:
	CARGO_PROFILE_RELEASE_LTO=off cargo build -p rten-cli --release

.PHONY: schema
schema: rten-model-file/src/schema_generated.rs rten-convert/rten_convert/schema_generated.py

.PHONY: clean
clean:
	rm -rf dist/*
	rm -rf target/

.PHONY: check
check: checkformatting test lint docs

.PHONY: checkformatting
checkformatting:
	cargo fmt --check

.PHONY: docs
docs:
	RUSTDOCFLAGS='-D warnings' cargo doc -p rten --features mmap,random

.PHONY: lint
lint:
	# `-D warnings` triggers non-zero exit on warnings. `-A unknown_lints` allows
	# for compiling with an older Rust version (eg. MSRV) that doesn't know about
	# some lints referenced in config/attributes.
	cargo clippy --workspace -- -D warnings -A unknown_lints

.PHONY: miri
miri:
	# - Only the tensor lib is currently tested. Testing the main crate will
	#   require changes to prevent tests taking too long to run.
	cargo +nightly miri test -p rten-tensor

# Run tests for all crates with all features enabled that do not require
# nightly Rust.
.PHONY: test
test:
	cargo test --no-fail-fast --workspace --features rten/all-ops,rten/mmap,rten-generate/text-decoder,rten-tensor/serde,rten-serialize/npy,rten-serialize/npz

# Cargo profile used to build the Python bindings. Defaults to release because
# debug builds are too slow for most uses.
PY_PROFILE ?= release

# Build the Python bindings.
#
# `--reinstall-package rten` forces the extension module to be rebuilt if the
# Rust sources changed.
.PHONY: build-py
build-py:
	cd rten-py && MATURIN_PEP517_ARGS="--profile $(PY_PROFILE)" uv sync --reinstall-package rten

# Build the Python bindings and run their tests.
.PHONY: test-py
test-py: PY_PROFILE = dev
test-py: build-py
	cd rten-py && uv run --no-sync pytest tests

# Default to running tests for the main crate unless otherwise specified.
PACKAGE ?= rten
CRATE := $(subst -,_,$(PACKAGE))

# Support enabling WASM relaxed-simd extension by passing `RELAXED=y`.
ifeq ($(RELAXED),y)
WASM_TARGET_FEATURES := -C target-feature=+simd128,+relaxed-simd
else
WASM_TARGET_FEATURES := -C target-feature=+simd128
endif

.PHONY: wasm
wasm:
	RUSTFLAGS="$(WASM_TARGET_FEATURES)" cargo build --features=wasm_api --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --target web --weak-refs
	# This makes the binary smaller but also removes all symbols. Comment this
	# out to get a release WASM build with symbols.
	tools/optimize-wasm.sh dist/rten_bg.wasm

.PHONY: wasm-nosimd
wasm-nosimd:
	cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --out-name rten-nosimd --target web --weak-refs
	tools/optimize-wasm.sh dist/rten-nosimd_bg.wasm

.PHONY: wasm-all
wasm-all: wasm wasm-nosimd

.PHONY: wasm-docs
wasm-docs:
	RUSTDOCFLAGS="-D warnings $(WASM_TARGET_FEATURES)" cargo doc -p rten-simd --target wasm32-unknown-unknown

# Run wasm tests using `make wasm-test PACKAGE={package_name}`
#
# WASM tests run with `--nocapture` as otherwise assertion failure panic messages
# are not printed if a test assert fails.
.PHONY: wasm-test
wasm-test:
	rm -f target/wasm32-wasi/debug/deps/$(CRATE)-*.wasm
	RUSTFLAGS="$(WASM_TARGET_FEATURES)" cargo build --target wasm32-wasip1 --tests -p $(PACKAGE)
	wasmtime --dir . target/wasm32-wasip1/debug/deps/${CRATE}-*.wasm --nocapture

# Run wasm benchmark using `make wasm-test PACKAGE={package_name} BENCH={bench_name}`
.PHONY: wasm-bench
wasm-bench:
	rm -f target/wasm32-wasi/release/deps/$(CRATE)-*.wasm
	RUSTFLAGS="$(WASM_TARGET_FEATURES)" cargo build --target wasm32-wasip1 --tests -p $(PACKAGE) -r
	wasmtime --dir . target/wasm32-wasip1/release/deps/$(CRATE)-*.wasm --nocapture --ignored $(BENCH)

MODEL_SCHEMA=rten-model-file/src/schema.fbs

rten-model-file/src/schema_generated.rs: $(MODEL_SCHEMA)
	flatc -o rten-model-file/src/ --rust $(MODEL_SCHEMA)
	cargo fmt

rten-convert/rten_convert/schema_generated.py: $(MODEL_SCHEMA)
	flatc -o rten-convert/rten_convert --gen-onefile --gen-object-api --python $(MODEL_SCHEMA)

.PHONY: gen-pytorch-references
gen-pytorch-references:
	python -m pytorch-ref-tests.rnn
