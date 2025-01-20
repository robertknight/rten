.PHONY: all
all:
	cargo build --workspace

.PHONY: schema
schema: src/schema_generated.rs rten-convert/rten_convert/schema_generated.py

.PHONY: clean
clean:
	rm -rf dist/*
	rm -rf target/

.PHONY: check
check: checkformatting test lint

.PHONY: checkformatting
checkformatting:
	cargo fmt --check

.PHONY: docs
docs:
	RUSTDOCFLAGS='-D warnings' cargo doc -p rten --features mmap,random

.PHONY: lint
lint:
	cargo clippy --workspace

.PHONY: miri
miri:
	# - Only the tensor lib is currently tested. Testing the main crate will
	#   require changes to prevent tests taking too long to run.
	cargo +nightly miri test -p rten-tensor

# Run tests for all crates with all features enabled that do not require
# nightly Rust.
.PHONY: test
test:
	cargo test --workspace --features mmap,random,text-decoder,serde

.PHONY: wasm
wasm:
	RUSTFLAGS="-C target-feature=+simd128" cargo build --features=wasm_api --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --target web --weak-refs
	# This makes the binary smaller but also removes all symbols. Comment this
	# out to get a release WASM build with symbols.
	tools/optimize-wasm.sh dist/rten_bg.wasm

.PHONY: wasm-relaxedsimd
wasm-relaxedsimd:
	RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build --features=wasm_api --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --target web --weak-refs

.PHONY: wasm-nosimd
wasm-nosimd:
	cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --out-name rten-nosimd --target web --weak-refs
	tools/optimize-wasm.sh dist/rten-nosimd_bg.wasm

.PHONY: wasm-all
wasm-all: wasm wasm-nosimd

# WASM tests run with `--nocapture` as otherwise assertion failure panic messages
# are not printed if a test assert fails.
.PHONY: wasm-tests
wasm-tests:
	rm -f target/wasm32-wasi/debug/deps/rten-*.wasm
	RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-wasip1 --tests -p rten
	wasmtime --dir . target/wasm32-wasip1/debug/deps/rten-*.wasm --nocapture

.PHONY: wasm-tests
wasm-tests-simd:
	rm -f target/wasm32-wasi/debug/deps/rten_simd-*.wasm
	RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-wasip1 --tests -p rten-simd
	wasmtime --dir . target/wasm32-wasip1/debug/deps/rten_simd-*.wasm --nocapture

src/schema_generated.rs: src/schema.fbs
	flatc -o src/ --rust src/schema.fbs
	cargo fmt
	(echo "#![allow(clippy::all)]" && cat src/schema_generated.rs) > src/schema_generated.rs.tmp
	mv src/schema_generated.rs.tmp src/schema_generated.rs

rten-convert/rten_convert/schema_generated.py: src/schema.fbs
	flatc -o rten-convert/rten_convert --gen-onefile --gen-object-api --python src/schema.fbs


.PHONY: gen-pytorch-references
gen-pytorch-references:
	python -m pytorch-ref-tests.rnn
