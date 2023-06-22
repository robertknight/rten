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
	cargo clippy -- -Aclippy::needless_range_loop -Aclippy::too_many_arguments -Aclippy::derivable_impls -Aclippy::manual_memcpy -Aclippy::assertions_on_constants -Aclippy::uninlined_format_args

.PHONY: test
test:
	cargo test --workspace

.PHONY: wasm
wasm:
	RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/wasnn.wasm --out-dir dist/ --target web --weak-refs
	tools/optimize-wasm.sh dist/wasnn_bg.wasm

.PHONY: wasm-nosimd
wasm-nosimd:
	cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/release/wasnn.wasm --out-dir dist/ --out-name wasnn-nosimd --target web --weak-refs
	tools/optimize-wasm.sh dist/wasnn-nosimd_bg.wasm

.PHONY: wasm-all
wasm-all: wasm wasm-nosimd

src/schema_generated.rs: src/schema.fbs
	flatc -o src/ --rust src/schema.fbs
	cargo fmt
	(echo "#![allow(clippy::all)]" && cat src/schema_generated.rs) > src/schema_generated.rs.tmp
	mv src/schema_generated.rs.tmp src/schema_generated.rs

tools/schema_generated.py: src/schema.fbs
	flatc -o tools/ --gen-onefile --gen-object-api --python src/schema.fbs


.PHONY: gen-pytorch-references
gen-pytorch-references:
	python -m pytorch-ref-tests.rnn
