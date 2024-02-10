.PHONY: all
all: src/schema_generated.rs tools/schema_generated.py

.PHONY: clean
clean:
	rm -rf dist/*
	rm -rf target/

.PHONY: check
check: checkformatting test lint

.PHONY: checkformatting
checkformatting:
	cargo fmt --check

.PHONY: lint
lint:
	cargo clippy --workspace

.PHONY: miri
miri:
	# - Stacked borrow checks are disabled because they don't like tests involving
	#   non-overlapping mutable views of the same underlying buffer. Fixing this
	#   will probably involve changes to view internals.
	# - Only the tensor lib is currently tested. Testing the main crate will
	#   require changes to prevent tests taking too long to run.
	MIRIFLAGS="-Zmiri-disable-stacked-borrows" cargo +nightly miri test -p rten-tensor

.PHONY: test
test:
	cargo test --workspace

.PHONY: wasm
wasm:
	RUSTFLAGS="-C target-feature=+simd128" cargo build --features=wasm_api --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --target web --weak-refs
	tools/optimize-wasm.sh dist/rten_bg.wasm

.PHONY: wasm-nosimd
wasm-nosimd:
	cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --out-name rten-nosimd --target web --weak-refs
	tools/optimize-wasm.sh dist/rten-nosimd_bg.wasm

.PHONY: wasm-all
wasm-all: wasm wasm-nosimd

src/schema_generated.rs: src/schema.fbs
	flatc -o src/ --rust src/schema.fbs
	cargo fmt
	(echo "#![allow(clippy::all)]" && cat src/schema_generated.rs) > src/schema_generated.rs.tmp
	mv src/schema_generated.rs.tmp src/schema_generated.rs

rten-convert/rten_convert/schema_generated.py: src/schema.fbs
	flatc -o tools/ --gen-onefile --gen-object-api --python src/schema.fbs


.PHONY: gen-pytorch-references
gen-pytorch-references:
	python -m pytorch-ref-tests.rnn
