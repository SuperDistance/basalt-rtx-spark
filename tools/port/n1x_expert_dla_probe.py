# SPDX-License-Identifier: MIT
"""Compare a Flash-Next-shaped NVFP4 expert on the N1X DLA and Basalt CPU kernels."""

import argparse
from contextlib import ExitStack
import ctypes
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


MATRIX_SHAPES = ((HIDDEN, EMBEDDING), (HIDDEN, EMBEDDING), (EMBEDDING, HIDDEN))
MATRIX_BYTES = tuple(rows * columns // 64 * 36 for rows, columns in MATRIX_SHAPES)
BLOB_BYTES = sum(MATRIX_BYTES) + 16


def synthetic_blob(rng):
    packed = []
    for rows, columns in MATRIX_SHAPES:
        blocks = rng.integers(0, 256, size=(rows, columns // 64, 36), dtype=np.uint8)
        blocks[..., :4] = 0x10   # UE4M3 scale 1/32.
        packed.append(blocks.tobytes())
    return b"".join(packed) + struct.pack("<4f", 1, 1, 1, 0)


def model_blob(path, layer, expert):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_file import ModelFile

    model = ModelFile(path)
    geometry = json.loads(model.read("meta"))["gguf_kv"]
    if (geometry.get("qwen4exp.embedding_length"), geometry.get("qwen4exp.expert_feed_forward_length")) != (
        EMBEDDING, HIDDEN
    ):
        raise ValueError("The pack must use Flash-Next expert dimensions")
    if not 0 <= expert < geometry["qwen4exp.expert_count"]:
        raise ValueError(f"Expert {expert} is outside the pack")
    for line in model.read("experts.layout").decode("ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        layer_id, gate, down, layer_offset, stride, *up = map(int, line.split())
        if layer_id != layer:
            continue
        if gate != 40 or down != 40 or (up and up != [40]) or stride != BLOB_BYTES:
            raise ValueError(f"Layer {layer} does not have three native NVFP4 matrices")
        offset, section_bytes = model.section("experts")
        start = layer_offset + expert * stride
        if start + stride > section_bytes:
            raise ValueError("Expert falls outside the packed section")
        with path.open("rb") as stream:
            stream.seek(offset + start)
            return stream.read(stride)
    raise ValueError(f"Layer {layer} is missing from the expert layout")


def dequantize(blob):
    matrix_offset = 0
    scales = struct.unpack_from("<3f", blob, sum(MATRIX_BYTES))
    matrices = []
    for (rows, columns), size, tensor_scale in zip(MATRIX_SHAPES, MATRIX_BYTES, scales):
        blocks = np.frombuffer(blob, np.uint8, count=size, offset=matrix_offset).reshape(rows, columns // 64, 36)
        ue4m3 = blocks[..., :4] & 0x7F
        exp, mantissa = (ue4m3 >> 3).astype(np.int32), (ue4m3 & 7).astype(np.float32)
        scale = np.where(exp == 0, mantissa / 1024, np.ldexp(1 + mantissa / 8, exp - 8))
        scale[ue4m3 == 0x7F] = 0
        codes = blocks[..., 4:].reshape(rows, columns // 64, 4, 8)
        values = np.concatenate((FP4[codes & 15], FP4[codes >> 4]), axis=-1)
        matrices.append((values * (scale * tensor_scale)[..., None]).reshape(rows, columns))
        matrix_offset += size
    return matrices


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


def gpu_comparison(path, matrices, module, data, reference):
    lib = ctypes.CDLL(str(path))
    lib.n1x_expert_gpu_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.n1x_expert_gpu_destroy.argtypes = [ctypes.c_void_p]
    lib.n1x_expert_gpu_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 8
    cuda = module.dev.cuda
    with ExitStack() as cleanup:
        def allocate(size):
            ptr = cuda.alloc(size)
            cleanup.callback(cuda.free, ptr)
            return ptr

        x, y = allocate(data.nbytes), allocate(data.nbytes)
        weights = []
        for matrix in matrices:
            packed = matrix.astype(np.float16).tobytes()
            ptr = allocate(len(packed))
            cuda.to_device(ptr, packed)
            weights.append(ptr)
        tokens = data.shape[-1]
        intermediate = [allocate(tokens * HIDDEN * 2) for _ in range(3)]
        stream = cuda.stream()
        cuda.lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
        cleanup.callback(cuda.lib.cuStreamDestroy_v2, stream)
        handle = ctypes.c_void_p()
        code = lib.n1x_expert_gpu_create(ctypes.byref(handle), stream)
        if code != 0:
            raise RuntimeError(f"cuBLAS initialization failed ({code})")
        cleanup.callback(lib.n1x_expert_gpu_destroy, handle)
        cleanup.callback(cuda.sync, stream)
        cuda.to_device(x, data.tobytes())

        def gpu_submit():
            code = lib.n1x_expert_gpu_run(handle, stream, tokens, x, *weights, *intermediate, y)
            if code != 0:
                raise RuntimeError(f"CUDA or cuBLAS expert submission failed ({code})")

        def gpu_run():
            gpu_submit()
            cuda.sync(stream)

        def both_run():
            gpu_submit()
            module.submit()
            cuda.sync(stream)
            module.wait()

        gpu_run()
        output = np.frombuffer(cuda.from_device(y, data.nbytes), np.float16).reshape(EMBEDDING, -1).T
        error = relative_rms(output.astype(np.float32), reference)
        if not error < 0.01:
            raise RuntimeError(f"GPU output failed validation (relative RMS {error:.6g})")
        return {"relative_rms_error": error, "gpu_submit_wait": timings(gpu_run, 64),
                "same_context_gpu_and_dla_submit_wait": timings(both_run, 64)}


def measure(args, npu_dla):
    rng = np.random.default_rng(121)
    payload = model_blob(args.model, args.layer, args.expert) if args.model else synthetic_blob(rng)
    matrices = dequantize(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    blob = args.output / "expert-nvfp4.bin"
    blob.write_bytes(payload)
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
            graph = expert_graph(tokens, matrices)
            loadable = npu_dla.compile_loadable(graph)
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
                    if args.gpu_library:
                        try:
                            result["gpu"] = gpu_comparison(args.gpu_library, matrices, module, data, reference)
                        except (RuntimeError, OSError) as exc:
                            result["gpu_error"] = str(exc)
                finally:
                    module.close()
        except RuntimeError as exc:
            result["dla_error"] = str(exc)
        results.append(result)
        print(json.dumps(result), flush=True)
    source = {"pack": args.model.name, "layer": args.layer, "expert": args.expert} if args.model else "synthetic"
    return {"geometry": [EMBEDDING, HIDDEN, EMBEDDING], "tensorrt": npu_dla.tensorrt().__version__, "source": source,
            "weight_bytes_cpu": blob.stat().st_size, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True, help="Directory containing npu_common and npu_dla")
    parser.add_argument("--cudla-bin", type=Path, required=True)
    parser.add_argument("--dll-dir", type=Path, action="append", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 32, 64])
    parser.add_argument("--cpu-workers", type=int, default=14)
    parser.add_argument("--model", type=Path, help="Use a native NVFP4 Flash-Next pack for the weights")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--gpu-library", type=Path, help="DLL built from n1x_expert_gpu_probe.cu for a cuBLAS comparison")
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
