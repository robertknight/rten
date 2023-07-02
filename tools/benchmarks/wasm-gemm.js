#!/usr/bin/env node

// Matrix multiplication performance test script for WebAssembly.
//
// The matmul implementation in the Rust crate has a similar set of tests for
// the native environment.

import { readFileSync } from "node:fs";

// TensorFlow.js dependencies. These will need to be installed separately via
// npm before you can run this script.
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";

import { Tensor, initSync } from "../../index.js";

// Init ML libs
await tf.setBackend("wasm");

const wasmBin = readFileSync("dist/wasnn_bg.wasm");
initSync(wasmBin);

// Run tests
const cases = [
  { m: 512, n: 512, k: 512 }, // Square
  { m: 128, n: 2048, k: 512 }, // Wide
  { m: 2048, n: 128, k: 512 }, // Tall
];

function logResult(engine, elapsedMs, m, n, k, iters) {
  const elapsedSecs = elapsedMs / 1000.0;
  const flops = (2 * m * n * k * iters) / elapsedSecs;
  const gflops = flops / 10 ** 9;

  const round = (f) => f.toFixed(2);

  console.log(
    `engine ${engine} m ${m} n ${n} k ${k} iters ${iters}. Duration ${round(
      elapsedMs
    )}ms (${round(elapsedMs / iters)} ms/iter). GFLOPS ${round(gflops)}`
  );
}

function timeIt(iters, callback) {
  const start = performance.now();
  for (let i = 0; i < iters; i++) {
    callback();
  }
  const end = performance.now();
  return end - start;
}

/**
 * Run a benchmark of `iters` iterations of matrix multiplication using random
 * inputs of size `[M, K]` and `[K, N]`.
 */
function testWasnnMatmul(m, n, k, iters) {
  const seedA = 1234n;
  const seedB = 4567n;
  const a = Tensor.rand([m, k], seedA);
  const b = Tensor.rand([k, n], seedB);

  const elapsed = timeIt(iters, () => {
    const c = a.matmul(b);

    // Free the output immediately so the memory can be re-used in the next
    // iteration.
    c.free();
  });

  logResult("Wasnn", elapsed, m, n, k, iters);
}

/**
 * Run a benchmark of `iters` iterations of matrix multiplication using random
 * inputs of size `[M, K]` and `[K, N]`.
 */
function testTensorflowMatMul(m, n, k, iters) {
  const a = tf.randomUniform([m, k]);
  const b = tf.randomUniform([k, n]);

  const elapsed = timeIt(iters, () => {
    const c = tf.matMul(a, b);

    // Free the output immediately so the memory can be re-used in the next
    // iteration.
    c.dispose();
  });

  logResult("TF.js", elapsed, m, n, k, iters);
}

for (const { m, n, k } of cases) {
  const iters = 100;

  testWasnnMatmul(m, n, k, iters);
  testTensorflowMatMul(m, n, k, iters);
}
