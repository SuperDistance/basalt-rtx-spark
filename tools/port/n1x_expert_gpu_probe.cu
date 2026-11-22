// SPDX-License-Identifier: MIT
// FP16 cuBLAS reference for the same expert and CUDA context used by the DLA probe.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int embedding = 2560;
constexpr int intermediate = 640;

__global__ void swiglu(const __half* gate, const __half* up, __half* hidden, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float g = __half2float(gate[i]);
        hidden[i] = __float2half_rn(g / (1.f + expf(-g)) * __half2float(up[i]));
    }
}

cublasStatus_t project(cublasHandle_t handle, int tokens, int in, int out, const void* input,
                       const void* weights, void* output) {
    // NCHW tensors at H=1 are [channels, tokens] with tokens contiguous.
    // Compute the transpose as a column-major [tokens, out] product.
    const float one = 1.f, zero = 0.f;
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, tokens, out, in, &one,
                        input, CUDA_R_16F, tokens, weights, CUDA_R_16F, in, &zero, output, CUDA_R_16F,
                        tokens, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // namespace

extern "C" __declspec(dllexport) int n1x_expert_gpu_create(cublasHandle_t* handle, cudaStream_t stream) {
    if (const auto rc = cublasCreate(handle); rc != CUBLAS_STATUS_SUCCESS) return 1000 + rc;
    if (const auto rc = cublasSetStream(*handle, stream); rc != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(*handle);
        *handle = nullptr;
        return 1000 + rc;
    }
    return 0;
}

extern "C" __declspec(dllexport) int n1x_expert_gpu_destroy(cublasHandle_t handle) {
    return cublasDestroy(handle);
}

extern "C" __declspec(dllexport) int n1x_expert_gpu_run(
    cublasHandle_t handle, cudaStream_t stream, int tokens, const void* input, const void* wg, const void* wu,
    const void* wd, __half* gate, __half* up, __half* hidden, void* output) {
    if (const auto rc = project(handle, tokens, embedding, intermediate, input, wg, gate); rc != CUBLAS_STATUS_SUCCESS)
        return 1000 + rc;
    if (const auto rc = project(handle, tokens, embedding, intermediate, input, wu, up); rc != CUBLAS_STATUS_SUCCESS)
        return 1000 + rc;
    const int count = tokens * intermediate;
    swiglu<<<(count + 255) / 256, 256, 0, stream>>>(gate, up, hidden, count);
    if (const auto rc = cudaGetLastError(); rc != cudaSuccess) return rc;
    const auto rc = project(handle, tokens, intermediate, embedding, hidden, wd, output);
    return rc == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + rc;
}
