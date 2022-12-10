export {
  default as init,
  initSync,
  Model,
  Tensor,
  TensorList,
} from "./dist/wasnn.js";

import { TensorList } from "./dist/wasnn.js";

// Make TensorList usable with `Array.from`, `for ... of` etc.
TensorList.prototype[Symbol.iterator] = function* () {
  for (let i = 0; i < this.length; i++) {
    yield this.item(i);
  }
};

/**
 * Construct a TensorList from an iterable, such as an Array, Set etc.
 */
TensorList.from = (iterable) => {
  const list = new TensorList();
  for (const tensor of iterable) {
    list.push(tensor);
  }
  return list;
};

/**
 * Return true if the current JS environment supports the SIMD extension for
 * WebAssembly.
 */
function simdSupported() {
  // Tiny WebAssembly file generated from the following source using `wat2wasm`:
  //
  // (module
  //   (func (result v128)
  //     i32.const 0
  //     i8x16.splat
  //     i8x16.popcnt
  //   )
  // )
  const simdTest = Uint8Array.from([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1,
    8, 0, 65, 0, 253, 15, 253, 98, 11,
  ]);
  return WebAssembly.validate(simdTest);
}

/**
 * Return the filename of the preferred Wasnn binary for the current
 * environment.
 */
export function binaryName() {
  if (simdSupported()) {
    return "wasnn_bg.wasm";
  } else {
    return "wasnn-nosimd_bg.wasm";
  }
}
