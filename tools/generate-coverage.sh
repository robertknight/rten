#!/bin/sh

# See https://blog.rng0.io/how-to-do-code-coverage-in-rust and
# https://github.com/mozilla/grcov.

rm -rf target/coverage

CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' LLVM_PROFILE_FILE='cargo-test-%p-%m.profraw' cargo test
grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o target/coverage/html
rm cargo-test-*.profraw
