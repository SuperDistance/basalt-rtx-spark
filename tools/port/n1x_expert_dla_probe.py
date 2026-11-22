# SPDX-License-Identifier: MIT
"""Compare a synthetic Flash-Next-shaped expert on the N1X DLA and Basalt CPU kernels."""

import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EMBEDDING = 2560
HIDDEN = 640
FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def synthetic_weights(rng, output, inputs):
    # UE4M3 0x10 is 1/32, or 1/64 with the doubled E2M1 lookup table.
    codes = rng.integers(0, 16, size=(output, inputs // 64, 4, 16), dtype=np.uint8)
    blocks = np.empty((output, inputs // 64, 36), dtype=np.uint8)
    blocks[..., :4] = 0x10
    blocks[..., 4:] = (codes[..., :8] | (codes[..., 8:] << 4)).reshape(output, inputs // 64, 32)
    matrix = (FP4[codes] / 64).reshape(output, inputs)
    return blocks.tobytes(), matrix


def expert_graph(tokens, matrices):
    nodes = [
        helper.make_node("Conv", ["x", "wg"], ["gate"], kernel_shape=[1, 1]),
        helper.make_node("Conv", ["x", "wu"], ["up"], kernel_shape=[1, 1]),
        helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
        helper.make_node("Mul", ["gate", "sigmoid"], ["silu"]),
        helper.make_node("Mul", ["silu", "up"], ["hidden"]),
        helper.make_node("Conv", ["hidden", "wd"], ["y"], kernel_shape=[1, 1]),
    ]
    shape = (1, EMBEDDING, 1, tokens)
    graph = helper.make_graph(
        nodes, "flash_next_expert",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)],
        [numpy_helper.from_array(matrix.astype(np.float16)[:, :, None, None], name)
         for matrix, name in zip(matrices, ("wg", "wu", "wd"))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def relative_rms(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def timings(run, samples):
    for _ in range(8):
        run()
    elapsed = []
    for _ in range(samples):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(elapsed),
            "p95_ms": statistics.quantiles(elapsed, n=20, method="inclusive")[18], "samples": samples}


def measure(args, npu_dla):
    rng = np.random.default_rng(121)
    packed, matrices = zip(*(synthetic_weights(rng, rows, columns)
                             for rows, columns in [(HIDDEN, EMBEDDING), (HIDDEN, EMBEDDING), (EMBEDDING, HIDDEN)]))
    args.output.mkdir(parents=True, exist_ok=True)
    blob = args.output / "synthetic-nvfp4.bin"
    blob.write_bytes(b"".join(packed) + struct.pack("<4f", 1, 1, 1, 0))
    results = []
    for tokens in args.tokens:
        fp16_input = rng.normal(0, 0.125, (tokens, EMBEDDING)).astype(np.float16)
        inputs = fp16_input.astype(np.float32)
        gate, up, down = matrices
        g = inputs @ gate.T
        reference = ((g / (1 + np.exp(-g))) * (inputs @ up.T)) @ down.T
        x, y = args.output / f"input-{tokens}.f32", args.output / f"output-{tokens}.f32"
        inputs.tofile(x)
        completed = subprocess.run([args.cpu, blob, x, str(tokens), y, str(args.cpu_workers)],
                                   capture_output=True, text=True, check=True)
        cpu = json.loads(completed.stdout)
        cpu["relative_rms_error"] = relative_rms(np.fromfile(y, np.float32).reshape(reference.shape), reference)
        if not cpu["relative_rms_error"] < 0.04:
            raise RuntimeError(f"CPU output failed validation at {tokens} tokens")
        result = {"tokens_for_one_expert": tokens, "cpu": cpu}
        try:
            loadable = npu_dla.compile_loadable(expert_graph(tokens, matrices))
            result["dla_loadable_bytes"] = len(loadable)
            with npu_dla.DlaDevice() as device:
                module = device.load(loadable)
                try:
                    data = np.ascontiguousarray(fp16_input.T.reshape(1, EMBEDDING, 1, tokens))
                    module.write(0, data)
                    module.run()
                    output = module.read().reshape(EMBEDDING, tokens).T.astype(np.float32)
                    error = relative_rms(output, reference)
                    result["dla_relative_rms_error"] = error
                    if not error < 0.01:
                        raise RuntimeError("DLA output failed validation")
                    result["dla_submit_wait"] = timings(module.run, 64)

                    def roundtrip():
                        module.write(0, data)
                        module.run()
                        module.read()

                    result["dla_host_roundtrip"] = timings(roundtrip, 64)
                finally:
                    module.close()
        except RuntimeError as exc:
            result["dla_error"] = str(exc)
        results.append(result)
        print(json.dumps(result), flush=True)
    return {"geometry": [EMBEDDING, HIDDEN, EMBEDDING], "tensorrt": npu_dla.tensorrt().__version__,
            "weight_bytes_cpu": blob.stat().st_size, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True, help="Directory containing npu_common and npu_dla")
    parser.add_argument("--cudla-bin", type=Path, required=True)
    parser.add_argument("--dll-dir", type=Path, action="append", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument("--cpu-workers", type=int, default=14)
    args = parser.parse_args()
    dll_dirs = [args.cudla_bin, *args.dll_dir]
    dll_handles = [os.add_dll_directory(str(path)) for path in dll_dirs]
    os.environ["PATH"] = os.pathsep.join([*(str(path) for path in dll_dirs), os.environ["PATH"]])
    sys.path.insert(0, str(args.helpers))
    import npu_common
    import npu_dla
    import tensorrt

    npu_common.CUDLA_BIN = args.cudla_bin
    npu_dla._trt = tensorrt
    try:
        result = measure(args, npu_dla)
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    main()
