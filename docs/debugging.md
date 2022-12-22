# Debugging

This document provides strategies for debugging incorrect/different outputs in
Wasnn compared to other runtimes.

## Inspecting models

[Netron](https://netron.app) is an online tool, also available as an Electron
app for visualizing ONNX models.

## Comparing against ONNX Runtime

[ONNX Runtime](https://onnxruntime.ai) ("ORT") is the most mature implementation
of the ONNX specification, and is often used as a reference for correctness and
performance testing.

The general steps to use ORT to compare and debug unexpected output from Wasnn
are:

1. Create a Python script to execute the model with ORT, and a
   corresponding Rust binary to execute the model with Wasnn.

2. Verify that the model produces the expected results with ORT.

3. Verify that the inputs to the model, after all preprocessing and conversion
   to tensors, are the same in Wasnn and ORT.

4. Verify that there are significant differences in the Wasnn vs ORT output.

5. Compare the values of intermediate outputs in the graph to find where
   significant differences begin to arise. For small tensors, the values can
   simply be printed and inspected by eye. Most tensors will be larger however
   and so it is useful to get statistics of a comparison. A typical approach is:

   1. Run the model specifying an intermediate node as an output.

      Wasnn allows any node in the graph to be specified as an output. ORT on the
      other hand only allows nodes specified in the model's output list to be
      fetched as an output from a model run. The
      `tools/add-node-outputs-to-model.py` script works around this limitation of
      ORT by producing a modified ONNX model that lists every node in the graph
      in the output list.

   2. Write out the resulting intermediate tensors. The `Tensor::write` method
      can be used for this in Wasnn and the `write_tensor` function in
      `tools/debug_utils.py` in Python.

   3. Compare the resulting tensors. `tools/compare-tensors.py` compares tensor
      shapes and reports statistics on the absolute difference between
      corresponding values.

   Repeat steps 1-3 until you have identified where in the model discrepancies
   begin to arise. Note that very small differences for individual values,
   eg. on the order of 5 or 6 places after the decimal point, are normal for
   certain operations due to implementation differences.
