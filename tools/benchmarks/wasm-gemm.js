#!/usr/bin/env node

// Matrix multiplication performance test script for WebAssembly.
//
// The matmul implementation in the Rust crate has a similar set of tests for
// the native environment.

import { readFileSync } from "node:fs";

import { Tensor, initSync } from "../../index.js";

const wasmBin = readFileSync("dist/wasnn_bg.wasm");
initSync(wasmBin);

const cases = [
  { m: 512, n: 512, k: 512 }, // Square
  { m: 128, n: 2048, k: 512 }, // Wide
  { m: 2048, n: 128, k: 512 }, // Tall
];

for (const { m, n, k } of cases) {
  const iters = 100;

  const seedA = 1234n;
  const seedB = 4567n;
  const a = Tensor.rand([m, k], seedA);
  const b = Tensor.rand([k, n], seedB);

  const start = performance.now();
  for (let i = 0; i < iters; i++) {
    const c = a.matmul(b);

    // Free the output immediately so the memory can be re-used in the next
    // iteration.
    c.free();
  }
  const end = performance.now();
  const elapsedMs = end - start;
  const elapsedSecs = elapsedMs / 1000.0;

  const flops = (2 * m * n * k * iters) / elapsedSecs;
  const gflops = flops / 10 ** 9;

  const round = (f) => f.toFixed(2);

  console.log(
    `m ${m} n ${n} k ${k} iters ${iters}. Duration ${round(
      elapsedMs
    )}ms (${round(elapsedMs / iters)} ms/iter). GFLOPS ${round(gflops)}`
  );
}
