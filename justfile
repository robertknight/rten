# Set to "y" to enable the WASM relaxed-simd extension in WASM builds and tests.
relaxed := "n"

wasm_target_features := if relaxed == "y" {
  "-C target-feature=+simd128,+relaxed-simd"
} else {
  "-C target-feature=+simd128"
}

model_schema := "rten-model-file/src/schema.fbs"

# List available recipes.
_default:
    @just --list

# Build all crates in the workspace.
all:
    cargo build --workspace

# Build the CLI in release mode.
cli:
    cargo build -p rten-cli --release

# Build the CLI with ThinLTO disabled.
cli-no-lto:
    # Disabling ThinLTO makes the build faster but the binary will be larger and
    # slower. Performance is still usable though, unlike a debug build.
    CARGO_PROFILE_RELEASE_LTO=off cargo build -p rten-cli --release

# Generate Rust and Python code for reading the model schema.
schema:
    flatc -o rten-model-file/src/ --rust {{model_schema}}
    cargo fmt
    flatc -o rten-convert/rten_convert --gen-onefile --gen-object-api --python {{model_schema}}

# Remove build outputs.
clean:
    rm -rf dist/*
    rm -rf target/

# Run all checks that CI performs.
check: checkformatting test lint docs

# Check that source code is formatted correctly.
checkformatting:
    cargo fmt --check

# Build API docs, treating warnings as errors.
docs:
    RUSTDOCFLAGS='-D warnings' cargo doc -p rten --features mmap,random

# Lint all crates in the workspace.
lint:
    # `-D warnings` triggers non-zero exit on warnings. `-A unknown_lints` allows
    # for compiling with an older Rust version (eg. MSRV) that doesn't know about
    # some lints referenced in config/attributes.
    cargo clippy --workspace -- -D warnings -A unknown_lints

# Run tests under Miri.
miri:
    # - Only the tensor lib is currently tested. Testing the main crate will
    #   require changes to prevent tests taking too long to run.
    cargo +nightly miri test -p rten-tensor

# Run tests for all crates, with all features that work on stable Rust.
test:
    cargo test --no-fail-fast --workspace --features rten/all-ops,rten/mmap,rten-generate/text-decoder,rten-tensor/serde,rten-serialize/npy,rten-serialize/npz

# Build the WASM library and JS bindings.
wasm:
    RUSTFLAGS="{{wasm_target_features}}" cargo build --features=wasm_api --release --target wasm32-unknown-unknown
    wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --target web --weak-refs
    # This makes the binary smaller but also removes all symbols. Comment this
    # out to get a release WASM build with symbols.
    tools/optimize-wasm.sh dist/rten_bg.wasm

# Build the WASM library without SIMD support.
wasm-nosimd:
    cargo build --release --target wasm32-unknown-unknown
    wasm-bindgen target/wasm32-unknown-unknown/release/rten.wasm --out-dir dist/ --out-name rten-nosimd --target web --weak-refs
    tools/optimize-wasm.sh dist/rten-nosimd_bg.wasm

# Build both the SIMD and non-SIMD WASM libraries.
wasm-all: wasm wasm-nosimd

# Build API docs for the WASM target.
wasm-docs:
    RUSTDOCFLAGS="-D warnings {{wasm_target_features}}" cargo doc -p rten-simd --target wasm32-unknown-unknown

# Run tests for a package under WASM. Use `just relaxed=y wasm-test` for relaxed-simd.
wasm-test package="rten":
    # WASM tests run with `--nocapture` as otherwise assertion failure panic
    # messages are not printed if a test assert fails.
    rm -f target/wasm32-wasip1/debug/deps/{{replace(package, "-", "_")}}-*.wasm
    RUSTFLAGS="{{wasm_target_features}}" cargo build --target wasm32-wasip1 --tests -p {{package}}
    wasmtime --dir . target/wasm32-wasip1/debug/deps/{{replace(package, "-", "_")}}-*.wasm --nocapture

# Run a benchmark for a package under WASM.
wasm-bench package="rten" bench="":
    rm -f target/wasm32-wasip1/release/deps/{{replace(package, "-", "_")}}-*.wasm
    RUSTFLAGS="{{wasm_target_features}}" cargo build --target wasm32-wasip1 --tests -p {{package}} -r
    wasmtime --dir . target/wasm32-wasip1/release/deps/{{replace(package, "-", "_")}}-*.wasm --nocapture --ignored {{bench}}

# Generate reference outputs for RNN tests using PyTorch.
gen-pytorch-references:
    python -m pytorch-ref-tests.rnn
