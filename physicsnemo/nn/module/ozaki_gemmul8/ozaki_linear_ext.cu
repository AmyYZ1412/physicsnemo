#include "gemmul8.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_fp8.h>

#include "self_hipify.hpp"
#include "template_type.hpp"
#include "common.hpp"
#include "template_math.hpp"
#include "table.hpp"
#include "find_max.hpp"
#include "mod.hpp"
#include "matmult.hpp"
#include "scaling.hpp"
#include "scaling_fast_real.hpp"
#include "scaling_accu_real.hpp"
#include "conv_hi2mid_real.hpp"
#include "inverse_scaling_real.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

cublasLtHandle_t get_cublaslt_handle() {
    static cublasLtHandle_t handle = nullptr;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        TORCH_CHECK(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
    });
    return handle;
}

struct WorkBufferCache {
    std::mutex mtx;
    std::unordered_map<int64_t, torch::Tensor> buffers;

    torch::Tensor get(
        int64_t device_index,
        cudaStream_t stream,
        int64_t lwork,
        const torch::TensorOptions &options) {
        const int64_t rounded = ((lwork + 262143) / 262144) * 262144;
        const int64_t stream_key = static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream) & 0x7fffffff);
        const int64_t key = (device_index << 48) ^ (stream_key << 20) ^ rounded;

        std::lock_guard<std::mutex> lock(mtx);
        auto it = buffers.find(key);
        if (it != buffers.end() && it->second.numel() >= lwork) {
            return it->second;
        }
        torch::Tensor buf = torch::empty({rounded}, options.dtype(torch::kUInt8));
        buffers[key] = buf;
        return buf;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        buffers.clear();
    }
};

WorkBufferCache &get_work_cache() {
    static WorkBufferCache cache;
    return cache;
}

void check_matmul_inputs(
    const torch::Tensor &a,
    const torch::Tensor &b,
    int64_t num_moduli) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat64, "a must be float64");
    TORCH_CHECK(b.scalar_type() == torch::kFloat64, "b must be float64");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b inner dimensions differ");
    TORCH_CHECK(num_moduli >= 2 && num_moduli <= 20, "num_moduli must be in [2, 20]");
}

void check_bias(
    const torch::Tensor &output,
    const c10::optional<torch::Tensor> &bias) {
    if (bias.has_value() && bias.value().defined()) {
        const torch::Tensor &bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.device() == output.device(), "bias must be on the same CUDA device as output");
        TORCH_CHECK(bias_tensor.scalar_type() == torch::kFloat64, "bias must be float64");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias_tensor.size(0) == output.size(1), "bias size must match output features");
    }
}

void check_limb_tensor(
    const char *name,
    const torch::Tensor &tensor,
    const c10::Device &device,
    c10::ArrayRef<int64_t> sizes) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same CUDA device");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.sizes() == sizes, name, " shape mismatch");
}

__device__ __forceinline__ double triple_value(
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    int64_t idx) {
    return static_cast<double>(hi[idx]) + static_cast<double>(mid[idx]) + static_cast<double>(lo[idx]);
}

__device__ __forceinline__ double triple_value_row_major(
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    int64_t row,
    int64_t col,
    int64_t ld) {
    return triple_value(hi, mid, lo, row * ld + col);
}

__device__ __forceinline__ void two_sum_float(float a, float b, float &s, float &e) {
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

template <int limbs>
__device__ __forceinline__ void expansion_add_float(float (&acc)[limbs], float value) {
    float q = value;
    float tmp[limbs];
    #pragma unroll
    for (int i = 0; i < limbs; ++i) {
        float s, e;
        two_sum_float(q, acc[i], s, e);
        tmp[i] = e;
        q = s;
    }
    float s, e;
    two_sum_float(tmp[limbs - 1], q, s, e);
    #pragma unroll
    for (int i = 0; i < limbs - 2; ++i) {
        acc[i] = tmp[i + 1];
    }
    acc[limbs - 2] = e;
    acc[limbs - 1] = s;
}

__global__ void triple_sum_dim0_kernel(
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    float *__restrict__ out_hi,
    float *__restrict__ out_mid,
    float *__restrict__ out_lo,
    int64_t rows,
    int64_t cols) {
    constexpr int limbs = 3;
    __shared__ float shared[256][limbs];
    const int64_t col = blockIdx.x;
    float local[limbs] = {0.0f, 0.0f, 0.0f};
    for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const int64_t idx = row * cols + col;
        expansion_add_float<limbs>(local, hi[idx]);
        expansion_add_float<limbs>(local, mid[idx]);
        expansion_add_float<limbs>(local, lo[idx]);
    }
    #pragma unroll
    for (int limb = 0; limb < limbs; ++limb) {
        shared[threadIdx.x][limb] = local[limb];
    }
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int limb = 0; limb < limbs; ++limb) {
                expansion_add_float<limbs>(local, shared[threadIdx.x + stride][limb]);
            }
            #pragma unroll
            for (int limb = 0; limb < limbs; ++limb) {
                shared[threadIdx.x][limb] = local[limb];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_hi[col] = local[limbs - 1];
        out_mid[col] = local[limbs - 2];
        out_lo[col] = local[limbs - 3];
    }
}

template <int num_moduli>
__global__ void triple_compute_sft_fast_rows_kernel(
    unsigned rows,
    unsigned cols,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int16_t *__restrict__ sft) {
    __shared__ double s_amax[256];
    __shared__ double s_sum[256];
    const unsigned row = blockIdx.x;
    double amax = 0.0;
    double sum = 0.0;
    for (unsigned col = threadIdx.x; col < cols; col += blockDim.x) {
        const double x = triple_value_row_major(hi, mid, lo, row, col, ld);
        const double ax = fabs(x);
        amax = fmax(amax, ax);
        sum = fma(x, x, sum);
    }
    s_amax[threadIdx.x] = amax;
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_amax[threadIdx.x] = fmax(s_amax[threadIdx.x], s_amax[threadIdx.x + stride]);
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && row < rows) {
        const int shift = real::fast::compute_sft<gemmul8::Backend::INT8, num_moduli>(
            s_amax[0], s_sum[0]);
        sft[row] = int16_t(-shift);
    }
}

__global__ void triple_compute_sft_extract_rows_kernel(
    unsigned rows,
    unsigned cols,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int16_t *__restrict__ sft) {
    __shared__ double s_amax[256];
    const unsigned row = blockIdx.x;
    double amax = 0.0;
    for (unsigned col = threadIdx.x; col < cols; col += blockDim.x) {
        const double x = triple_value_row_major(hi, mid, lo, row, col, ld);
        amax = fmax(amax, fabs(x));
    }
    s_amax[threadIdx.x] = amax;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_amax[threadIdx.x] = fmax(s_amax[threadIdx.x], s_amax[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && row < rows) {
        sft[row] = int16_t(maxUFP<gemmul8::Backend::INT8> - Tilogb<double>(s_amax[0]));
    }
}

__global__ void triple_compute_sft_extract_cols_kernel(
    unsigned rows,
    unsigned cols,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int16_t *__restrict__ sft) {
    __shared__ double s_amax[256];
    const unsigned col = blockIdx.x;
    double amax = 0.0;
    for (unsigned row = threadIdx.x; row < rows; row += blockDim.x) {
        const double x = triple_value(hi, mid, lo, static_cast<int64_t>(col) * ld + row);
        amax = fmax(amax, fabs(x));
    }
    s_amax[threadIdx.x] = amax;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_amax[threadIdx.x] = fmax(s_amax[threadIdx.x], s_amax[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && col < cols) {
        sft[col] = int16_t(maxUFP<gemmul8::Backend::INT8> - Tilogb<double>(s_amax[0]));
    }
}

__global__ void triple_extractA_upper_kernel(
    unsigned rows,
    unsigned cols,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int8_t *__restrict__ out_lo,
    size_t ld_lo,
    const int16_t *__restrict__ sft) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= ld_lo) {
        return;
    }
    const double x = (col < cols)
        ? triple_value_row_major(hi, mid, lo, row, col, ld)
        : 0.0;
    out_lo[static_cast<int64_t>(row) * ld_lo + col] =
        upperBound_lo<gemmul8::Backend::INT8, double>(x, sft[row]);
}

__global__ void triple_extractB_upper_kernel(
    unsigned rows,
    unsigned cols,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int8_t *__restrict__ out_lo,
    size_t ld_lo,
    const int16_t *__restrict__ sft) {
    const unsigned col = blockIdx.x;
    for (unsigned row = threadIdx.x; row < ld_lo; row += blockDim.x) {
        const double x = (row < rows)
            ? triple_value(hi, mid, lo, static_cast<int64_t>(col) * ld + row)
            : 0.0;
        out_lo[static_cast<int64_t>(col) * ld_lo + row] =
            upperBound_lo<gemmul8::Backend::INT8, double>(x, sft[col]);
    }
}

template <int num_moduli>
__global__ void triple_scalingA_kernel(
    unsigned rows,
    unsigned cols,
    size_t inc_lo,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int8_t *__restrict__ out_lo,
    size_t ld_lo,
    const int16_t *__restrict__ sft) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= ld_lo) {
        return;
    }
    const double x = (col < cols)
        ? triple_value_row_major(hi, mid, lo, row, col, ld)
        : 0.0;
    const int shift = -sft[row];
    auto scaled = trunc_scalbn<gemmul8::Backend::INT8, double, num_moduli>::run(x, shift);
    int8_t *__restrict__ out = out_lo + static_cast<int64_t>(row) * ld_lo + col;
    ModUnroll<num_moduli, decltype(scaled)>::run(out, inc_lo, scaled);
}

template <int num_moduli>
__global__ void triple_scalingB_fast_kernel(
    unsigned rows,
    unsigned cols,
    size_t inc_lo,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int8_t *__restrict__ out_lo,
    size_t ld_lo,
    int16_t *__restrict__ sft) {
    __shared__ double s_amax[256];
    __shared__ double s_sum[256];
    const unsigned col = blockIdx.x;
    double amax = 0.0;
    double sum = 0.0;
    for (unsigned row = threadIdx.x; row < rows; row += blockDim.x) {
        const double x = triple_value(hi, mid, lo, static_cast<int64_t>(col) * ld + row);
        const double ax = fabs(x);
        amax = fmax(amax, ax);
        sum = fma(x, x, sum);
    }
    s_amax[threadIdx.x] = amax;
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_amax[threadIdx.x] = fmax(s_amax[threadIdx.x], s_amax[threadIdx.x + stride]);
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    int shift = 0;
    if (threadIdx.x == 0) {
        shift = real::fast::compute_sft<gemmul8::Backend::INT8, num_moduli>(
            s_amax[0], s_sum[0]);
        sft[col] = int16_t(-shift);
    }
    __syncthreads();
    shift = -sft[col];
    for (unsigned row = threadIdx.x; row < ld_lo; row += blockDim.x) {
        const double x = (row < rows)
            ? triple_value(hi, mid, lo, static_cast<int64_t>(col) * ld + row)
            : 0.0;
        auto scaled = trunc_scalbn<gemmul8::Backend::INT8, double, num_moduli>::run(x, shift);
        int8_t *__restrict__ out = out_lo + static_cast<int64_t>(col) * ld_lo + row;
        ModUnroll<num_moduli, decltype(scaled)>::run(out, inc_lo, scaled);
    }
}

template <int num_moduli>
__global__ void triple_scalingB_kernel(
    unsigned rows,
    unsigned cols,
    size_t inc_lo,
    const float *__restrict__ hi,
    const float *__restrict__ mid,
    const float *__restrict__ lo,
    size_t ld,
    int8_t *__restrict__ out_lo,
    size_t ld_lo,
    const int16_t *__restrict__ sft) {
    const unsigned col = blockIdx.x;
    const int shift = -sft[col];
    for (unsigned row = threadIdx.x; row < ld_lo; row += blockDim.x) {
        const double x = (row < rows)
            ? triple_value(hi, mid, lo, static_cast<int64_t>(col) * ld + row)
            : 0.0;
        auto scaled = trunc_scalbn<gemmul8::Backend::INT8, double, num_moduli>::run(x, shift);
        int8_t *__restrict__ out = out_lo + static_cast<int64_t>(col) * ld_lo + row;
        ModUnroll<num_moduli, decltype(scaled)>::run(out, inc_lo, scaled);
    }
}

template <int num_moduli>
void triple_scaling_launch(
    cudaStream_t stream,
    size_t m,
    size_t n,
    size_t k,
    const float *A_hi,
    const float *A_mid,
    const float *A_lo_src,
    size_t lda,
    int8_t *A_lo,
    size_t lda_lo,
    size_t incA_lo,
    int16_t *sftA,
    const float *B_hi,
    const float *B_mid,
    const float *B_lo_src,
    size_t ldb,
    int8_t *B_lo,
    size_t ldb_lo,
    size_t incB_lo,
    int16_t *sftB) {
    constexpr int threads = 256;
    dim3 scale_a_grid(
        static_cast<unsigned>((m + 15) / 16),
        static_cast<unsigned>((lda_lo + 15) / 16));
    dim3 scale_a_threads(16, 16);
    triple_scalingA_kernel<num_moduli><<<scale_a_grid, scale_a_threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(k), incA_lo,
        A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, sftA);
    triple_scalingB_kernel<num_moduli><<<static_cast<unsigned>(n), threads, 0, stream>>>(
        static_cast<unsigned>(k), static_cast<unsigned>(n), incB_lo,
        B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, sftB);
}

template <int num_moduli>
void triple_fast_scaling_launch(
    cudaStream_t stream,
    size_t m,
    size_t n,
    size_t k,
    const float *A_hi,
    const float *A_mid,
    const float *A_lo_src,
    size_t lda,
    int8_t *A_lo,
    size_t lda_lo,
    size_t incA_lo,
    int16_t *sftA,
    const float *B_hi,
    const float *B_mid,
    const float *B_lo_src,
    size_t ldb,
    int8_t *B_lo,
    size_t ldb_lo,
    size_t incB_lo,
    int16_t *sftB) {
    constexpr int threads = 256;
    triple_compute_sft_fast_rows_kernel<num_moduli><<<static_cast<unsigned>(m), threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(k), A_hi, A_mid, A_lo_src, lda, sftA);
    dim3 scale_a_grid(
        static_cast<unsigned>((m + 15) / 16),
        static_cast<unsigned>((lda_lo + 15) / 16));
    dim3 scale_a_threads(16, 16);
    triple_scalingA_kernel<num_moduli><<<scale_a_grid, scale_a_threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(k), incA_lo,
        A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, sftA);
    triple_scalingB_fast_kernel<num_moduli><<<static_cast<unsigned>(n), threads, 0, stream>>>(
        static_cast<unsigned>(k), static_cast<unsigned>(n), incB_lo,
        B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, sftB);
}

void triple_scaling(
    cudaStream_t stream,
    unsigned num_moduli,
    size_t m,
    size_t n,
    size_t k,
    const float *A_hi,
    const float *A_mid,
    const float *A_lo_src,
    size_t lda,
    int8_t *A_lo,
    size_t lda_lo,
    size_t incA_lo,
    int16_t *sftA,
    const float *B_hi,
    const float *B_mid,
    const float *B_lo_src,
    size_t ldb,
    int8_t *B_lo,
    size_t ldb_lo,
    size_t incB_lo,
    int16_t *sftB) {
    switch (num_moduli) {
    case 2: triple_scaling_launch<2>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 3: triple_scaling_launch<3>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 4: triple_scaling_launch<4>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 5: triple_scaling_launch<5>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 6: triple_scaling_launch<6>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 7: triple_scaling_launch<7>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 8: triple_scaling_launch<8>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 9: triple_scaling_launch<9>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 10: triple_scaling_launch<10>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 11: triple_scaling_launch<11>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 12: triple_scaling_launch<12>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 13: triple_scaling_launch<13>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 14: triple_scaling_launch<14>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 15: triple_scaling_launch<15>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 16: triple_scaling_launch<16>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 17: triple_scaling_launch<17>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 18: triple_scaling_launch<18>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 19: triple_scaling_launch<19>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    case 20: triple_scaling_launch<20>(stream, m, n, k, A_hi, A_mid, A_lo_src, lda, A_lo, lda_lo, incA_lo, sftA, B_hi, B_mid, B_lo_src, ldb, B_lo, ldb_lo, incB_lo, sftB); break;
    default: TORCH_CHECK(false, "num_moduli must be in [2, 20]");
    }
}

template <int num_moduli>
void triple_update_shifts_from_high_launch(
    cudaStream_t stream,
    size_t m,
    size_t n,
    const int32_t *C_hi,
    size_t ldc_hi,
    int16_t *sftA,
    int16_t *sftB) {
    constexpr dim3 threads(TILE_DIM, TILE_DIM);
    real::accu::compute_sft_rowwise_kernel<num_moduli><<<
        static_cast<unsigned>((m + (TILE_DIM - 1)) / TILE_DIM), threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(n), C_hi, ldc_hi, sftA);
    real::accu::compute_sft_colwise_kernel<num_moduli><<<
        static_cast<unsigned>(n), threads_scaling, 0, stream>>>(
        static_cast<unsigned>(m), C_hi, ldc_hi, sftB);
}

void triple_update_shifts_from_high(
    cudaStream_t stream,
    unsigned num_moduli,
    size_t m,
    size_t n,
    const int32_t *C_hi,
    size_t ldc_hi,
    int16_t *sftA,
    int16_t *sftB) {
    switch (num_moduli) {
    case 2: triple_update_shifts_from_high_launch<2>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 3: triple_update_shifts_from_high_launch<3>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 4: triple_update_shifts_from_high_launch<4>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 5: triple_update_shifts_from_high_launch<5>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 6: triple_update_shifts_from_high_launch<6>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 7: triple_update_shifts_from_high_launch<7>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 8: triple_update_shifts_from_high_launch<8>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 9: triple_update_shifts_from_high_launch<9>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 10: triple_update_shifts_from_high_launch<10>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 11: triple_update_shifts_from_high_launch<11>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 12: triple_update_shifts_from_high_launch<12>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 13: triple_update_shifts_from_high_launch<13>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 14: triple_update_shifts_from_high_launch<14>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 15: triple_update_shifts_from_high_launch<15>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 16: triple_update_shifts_from_high_launch<16>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 17: triple_update_shifts_from_high_launch<17>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 18: triple_update_shifts_from_high_launch<18>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 19: triple_update_shifts_from_high_launch<19>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    case 20: triple_update_shifts_from_high_launch<20>(stream, m, n, C_hi, ldc_hi, sftA, sftB); break;
    default: TORCH_CHECK(false, "num_moduli must be in [2, 20]");
    }
}

template <int num_moduli, typename TP>
__global__ void triple_inverse_split_kernel(
    size_t m,
    size_t n,
    const int8_t *__restrict__ C_mid,
    size_t ldc_mid,
    size_t inc_mid,
    float *__restrict__ out_hi,
    float *__restrict__ out_mid,
    float *__restrict__ out_lo,
    size_t ldc,
    const int16_t *__restrict__ sftA,
    const int16_t *__restrict__ sftB,
    const float *__restrict__ bias_hi,
    const float *__restrict__ bias_mid,
    const float *__restrict__ bias_lo,
    bool has_bias,
    TP P,
    double invP) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t size = m * n;
    if (idx >= size) {
        return;
    }
    const size_t col = idx / m;
    const size_t row = idx - col * m;
    const size_t mem_idx = col * ldc_mid + row;
    double value = real::invscal_device<num_moduli, double, TP, int8_t>(
        inc_mid, C_mid + mem_idx, P, invP, sftA[row] + sftB[col]);
    if (has_bias) {
        value += static_cast<double>(bias_hi[row]) +
            static_cast<double>(bias_mid[row]) +
            static_cast<double>(bias_lo[row]);
    }
    const size_t out_idx = col * ldc + row;
    const float hi = static_cast<float>(value);
    const double r0 = value - static_cast<double>(hi);
    const float mid = static_cast<float>(r0);
    const double r1 = r0 - static_cast<double>(mid);
    out_hi[out_idx] = hi;
    out_mid[out_idx] = mid;
    out_lo[out_idx] = static_cast<float>(r1);
}

template <int num_moduli>
void triple_inverse_split_launch(
    cudaStream_t stream,
    size_t m,
    size_t n,
    const int8_t *C_mid,
    size_t ldc_mid,
    size_t inc_mid,
    float *out_hi,
    float *out_mid,
    float *out_lo,
    size_t ldc,
    const int16_t *sftA,
    const int16_t *sftB,
    const float *bias_hi,
    const float *bias_mid,
    const float *bias_lo,
    bool has_bias,
    unsigned num_moduli_runtime) {
    using TP = std::conditional_t<(num_moduli <= threshold<gemmul8::Backend::INT8>::P_is_double), double, double2>;
    const double invP = table::get_invP<gemmul8::Backend::INT8>(num_moduli_runtime);
    const TP P = table::get_P<gemmul8::Backend::INT8, TP>(num_moduli_runtime);
    const size_t size = m * n;
    const int threads = 128;
    const int blocks = static_cast<int>((size + threads - 1) / threads);
    triple_inverse_split_kernel<num_moduli, TP><<<blocks, threads, 0, stream>>>(
        m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc,
        sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, P, invP);
}

void triple_inverse_split(
    cudaStream_t stream,
    unsigned num_moduli,
    size_t m,
    size_t n,
    const int8_t *C_mid,
    size_t ldc_mid,
    size_t inc_mid,
    float *out_hi,
    float *out_mid,
    float *out_lo,
    size_t ldc,
    const int16_t *sftA,
    const int16_t *sftB,
    const float *bias_hi,
    const float *bias_mid,
    const float *bias_lo,
    bool has_bias) {
    switch (num_moduli) {
    case 2: triple_inverse_split_launch<2>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 3: triple_inverse_split_launch<3>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 4: triple_inverse_split_launch<4>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 5: triple_inverse_split_launch<5>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 6: triple_inverse_split_launch<6>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 7: triple_inverse_split_launch<7>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 8: triple_inverse_split_launch<8>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 9: triple_inverse_split_launch<9>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 10: triple_inverse_split_launch<10>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 11: triple_inverse_split_launch<11>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 12: triple_inverse_split_launch<12>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 13: triple_inverse_split_launch<13>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 14: triple_inverse_split_launch<14>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 15: triple_inverse_split_launch<15>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 16: triple_inverse_split_launch<16>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 17: triple_inverse_split_launch<17>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 18: triple_inverse_split_launch<18>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 19: triple_inverse_split_launch<19>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    case 20: triple_inverse_split_launch<20>(stream, m, n, C_mid, ldc_mid, inc_mid, out_hi, out_mid, out_lo, ldc, sftA, sftB, bias_hi, bias_mid, bias_lo, has_bias, num_moduli); break;
    default: TORCH_CHECK(false, "num_moduli must be in [2, 20]");
    }
}

torch::Tensor ozaki_matmul_2d_cuda_impl(
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_moduli,
    bool fastmode) {
    check_matmul_inputs(a, b, num_moduli);
    const c10::cuda::CUDAGuard device_guard(a.device());

    torch::Tensor a_c = a.contiguous();
    torch::Tensor b_c = b.contiguous();

    const int64_t rows = a_c.size(0);
    const int64_t inner = a_c.size(1);
    const int64_t cols = b_c.size(1);
    torch::Tensor output = torch::empty({rows, cols}, a.options());
    if (rows == 0 || cols == 0) {
        return output;
    }
    if (inner == 0) {
        output.zero_();
        return output;
    }

    const size_t m = static_cast<size_t>(cols);
    const size_t n = static_cast<size_t>(rows);
    const size_t k = static_cast<size_t>(inner);
    const double alpha = 1.0;
    const double beta = 0.0;

    const size_t lwork = gemmul8::workSize<false, gemmul8::Backend::INT8>(
        m, n, k, static_cast<unsigned>(num_moduli));
    torch::Tensor work = torch::empty(
        {static_cast<int64_t>(lwork)},
        a.options().dtype(torch::kUInt8));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());
    gemmul8::gemmLt<double, gemmul8::Backend::INT8>(
        get_cublaslt_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        b_c.data_ptr<double>(),
        m,
        a_c.data_ptr<double>(),
        k,
        &beta,
        output.data_ptr<double>(),
        m,
        static_cast<unsigned>(num_moduli),
        fastmode,
        work.data_ptr<unsigned char>(),
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        stream);
    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}

} // namespace

torch::Tensor ozaki_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_moduli,
    bool fastmode) {
    return ozaki_matmul_2d_cuda_impl(a, b, num_moduli, fastmode);
}

torch::Tensor ozaki_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t num_moduli,
    bool fastmode) {
    torch::Tensor output = ozaki_matmul_2d_cuda_impl(input, weight.transpose(0, 1), num_moduli, fastmode);
    check_bias(output, bias);
    if (bias.has_value() && bias.value().defined()) {
        output.add_(bias.value().contiguous());
    }
    return output;
}

std::vector<torch::Tensor> ozaki_linear_forward_triple_cuda(
    torch::Tensor input_hi,
    torch::Tensor input_mid,
    torch::Tensor input_lo,
    torch::Tensor weight_hi,
    torch::Tensor weight_mid,
    torch::Tensor weight_lo,
    c10::optional<torch::Tensor> bias_hi,
    c10::optional<torch::Tensor> bias_mid,
    c10::optional<torch::Tensor> bias_lo,
    int64_t num_moduli,
    bool fastmode) {
    TORCH_CHECK(input_hi.is_cuda(), "input_hi must be a CUDA tensor");
    TORCH_CHECK(input_hi.scalar_type() == torch::kFloat32, "input_hi must be float32");
    TORCH_CHECK(weight_hi.is_cuda(), "weight_hi must be a CUDA tensor");
    TORCH_CHECK(weight_hi.scalar_type() == torch::kFloat32, "weight_hi must be float32");
    TORCH_CHECK(input_hi.device() == weight_hi.device(), "input and weight limbs must share a device");
    TORCH_CHECK(input_hi.dim() == 2, "input limbs must be 2D");
    TORCH_CHECK(weight_hi.dim() == 2, "weight limbs must be 2D");
    TORCH_CHECK(input_hi.size(1) == weight_hi.size(1), "input last dimension must match weight in_features");
    const c10::cuda::CUDAGuard device_guard(input_hi.device());

    check_limb_tensor("input_mid", input_mid, input_hi.device(), input_hi.sizes());
    check_limb_tensor("input_lo", input_lo, input_hi.device(), input_hi.sizes());
    check_limb_tensor("weight_mid", weight_mid, weight_hi.device(), weight_hi.sizes());
    check_limb_tensor("weight_lo", weight_lo, weight_hi.device(), weight_hi.sizes());

    const bool has_bias =
        bias_hi.has_value() && bias_hi.value().defined() &&
        bias_mid.has_value() && bias_mid.value().defined() &&
        bias_lo.has_value() && bias_lo.value().defined();
    const bool any_bias =
        (bias_hi.has_value() && bias_hi.value().defined()) ||
        (bias_mid.has_value() && bias_mid.value().defined()) ||
        (bias_lo.has_value() && bias_lo.value().defined());
    TORCH_CHECK(has_bias == any_bias, "bias limbs must be all defined or all omitted");

    TORCH_CHECK(num_moduli >= 2 && num_moduli <= 20, "num_moduli must be in [2, 20]");
    TORCH_CHECK(!fastmode, "forward_triple does not yet support fastmode=True");

    if (has_bias) {
        const auto bias_sizes = c10::ArrayRef<int64_t>(bias_hi.value().sizes());
        TORCH_CHECK(bias_hi.value().dim() == 1, "bias limbs must be 1D");
        TORCH_CHECK(bias_hi.value().size(0) == weight_hi.size(0), "bias size must match output features");
        check_limb_tensor("bias_hi", bias_hi.value(), input_hi.device(), bias_sizes);
        check_limb_tensor("bias_mid", bias_mid.value(), input_hi.device(), bias_sizes);
        check_limb_tensor("bias_lo", bias_lo.value(), input_hi.device(), bias_sizes);
    }

    torch::Tensor input_hi_c = input_hi.contiguous();
    torch::Tensor input_mid_c = input_mid.contiguous();
    torch::Tensor input_lo_c = input_lo.contiguous();
    torch::Tensor weight_hi_c = weight_hi.contiguous();
    torch::Tensor weight_mid_c = weight_mid.contiguous();
    torch::Tensor weight_lo_c = weight_lo.contiguous();
    torch::Tensor bias_hi_c;
    torch::Tensor bias_mid_c;
    torch::Tensor bias_lo_c;
    if (has_bias) {
        bias_hi_c = bias_hi.value().contiguous();
        bias_mid_c = bias_mid.value().contiguous();
        bias_lo_c = bias_lo.value().contiguous();
    }

    const int64_t rows = input_hi_c.size(0);
    const int64_t inner = input_hi_c.size(1);
    const int64_t cols = weight_hi.size(0);
    auto out_options = input_hi_c.options().dtype(torch::kFloat32);
    torch::Tensor out_hi = torch::empty({rows, cols}, out_options);
    torch::Tensor out_mid = torch::empty_like(out_hi);
    torch::Tensor out_lo = torch::empty_like(out_hi);
    if (rows == 0 || cols == 0) {
        return {out_hi, out_mid, out_lo};
    }
    if (inner == 0) {
        out_hi.zero_();
        out_mid.zero_();
        out_lo.zero_();
        return {out_hi, out_mid, out_lo};
    }

    const size_t m = static_cast<size_t>(cols);
    const size_t n = static_cast<size_t>(rows);
    const size_t k = static_cast<size_t>(inner);
    const size_t lda_lo = padding(k);
    const size_t ldb_lo = lda_lo;
    const size_t ldc_hi = padding(m);
    const size_t sizeA = lda_lo * ldc_hi;
    const size_t sizeB = ldb_lo * n;
    const size_t sizeC = ldc_hi * n;
    const size_t sizeC_4 = sizeC >> 2;
    const size_t size_vecA = ldc_hi;
    const size_t size_vecB = padding(n);
    constexpr int32_t one = 1;
    constexpr int32_t zero = 0;

    const size_t lwork = gemmul8::workSize<false, gemmul8::Backend::INT8>(
        m, n, k, static_cast<unsigned>(num_moduli));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_hi_c.device().index());
    torch::Tensor work = get_work_cache().get(
        input_hi_c.device().index(),
        stream,
        static_cast<int64_t>(lwork),
        input_hi_c.options());
    void *work_aligned = align256(work.data_ptr<unsigned char>());
    const size_t offsetA = sizeA * table::num_mat<gemmul8::Backend::INT8>(static_cast<unsigned>(num_moduli));
    const size_t offsetB = sizeB * table::num_mat<gemmul8::Backend::INT8>(static_cast<unsigned>(num_moduli));
    int8_t *const A_lo = reinterpret_cast<int8_t *>(work_aligned);
    int16_t *const sftA = reinterpret_cast<int16_t *>(A_lo + offsetA);
    int8_t *const B_lo = reinterpret_cast<int8_t *>(sftA + size_vecA);
    int16_t *const sftB = reinterpret_cast<int16_t *>(B_lo + offsetB);
    int8_t *const C_mid = reinterpret_cast<int8_t *>(sftB + size_vecB);
    void *work_nativeGemm = reinterpret_cast<void *>(C_mid + (static_cast<unsigned>(num_moduli) - 1) * sizeC);
    int32_t *const C_hi = reinterpret_cast<int32_t *>(
        static_cast<int8_t *>(work_nativeGemm) +
        std::max(size_t(1) << 25, sizeof(int8_t) * sizeC));

    table::upload_constants<gemmul8::Backend::INT8>(stream);
    Handle_t handle(CublasLtTag{}, get_cublaslt_handle());
    set_handle<gemmul8::Backend::INT8>(
        stream, handle, static_cast<int>(ldc_hi), static_cast<int>(n), static_cast<int>(lda_lo),
        ldb_lo, ldb_lo, ldc_hi, work_nativeGemm, size_t(1) << 25);

    constexpr int threads = 256;
    triple_compute_sft_extract_rows_kernel<<<static_cast<unsigned>(m), threads, 0, stream>>>(
        static_cast<unsigned>(m),
        static_cast<unsigned>(k),
        weight_hi_c.data_ptr<float>(),
        weight_mid_c.data_ptr<float>(),
        weight_lo_c.data_ptr<float>(),
        k,
        sftA);
    triple_compute_sft_extract_cols_kernel<<<static_cast<unsigned>(n), threads, 0, stream>>>(
        static_cast<unsigned>(k),
        static_cast<unsigned>(n),
        input_hi_c.data_ptr<float>(),
        input_mid_c.data_ptr<float>(),
        input_lo_c.data_ptr<float>(),
        k,
        sftB);
    dim3 extract_a_grid(
        static_cast<unsigned>((m + 15) / 16),
        static_cast<unsigned>((lda_lo + 15) / 16));
    dim3 extract_a_threads(16, 16);
    triple_extractA_upper_kernel<<<extract_a_grid, extract_a_threads, 0, stream>>>(
        static_cast<unsigned>(m),
        static_cast<unsigned>(k),
        weight_hi_c.data_ptr<float>(),
        weight_mid_c.data_ptr<float>(),
        weight_lo_c.data_ptr<float>(),
        k,
        A_lo,
        lda_lo,
        sftA);
    triple_extractB_upper_kernel<<<static_cast<unsigned>(n), threads, 0, stream>>>(
        static_cast<unsigned>(k),
        static_cast<unsigned>(n),
        input_hi_c.data_ptr<float>(),
        input_mid_c.data_ptr<float>(),
        input_lo_c.data_ptr<float>(),
        k,
        B_lo,
        ldb_lo,
        sftB);
    C10_CUDA_CHECK(cudaGetLastError());
    gemm_low_prec_i8x1(
        stream,
        handle,
        static_cast<int>(ldc_hi),
        static_cast<int>(n),
        static_cast<int>(lda_lo),
        &one,
        A_lo,
        lda_lo,
        B_lo,
        ldb_lo,
        &zero,
        C_hi,
        ldc_hi);
    triple_update_shifts_from_high(
        stream,
        static_cast<unsigned>(num_moduli),
        m,
        n,
        C_hi,
        ldc_hi,
        sftA,
        sftB);
    C10_CUDA_CHECK(cudaGetLastError());

    triple_scaling(
        stream,
        static_cast<unsigned>(num_moduli),
        m,
        n,
        k,
        weight_hi_c.data_ptr<float>(),
        weight_mid_c.data_ptr<float>(),
        weight_lo_c.data_ptr<float>(),
        k,
        A_lo,
        lda_lo,
        sizeA,
        sftA,
        input_hi_c.data_ptr<float>(),
        input_mid_c.data_ptr<float>(),
        input_lo_c.data_ptr<float>(),
        k,
        B_lo,
        ldb_lo,
        sizeB,
        sftB);
    C10_CUDA_CHECK(cudaGetLastError());

    int8_t *A_lo_tmp = A_lo;
    int8_t *B_lo_tmp = B_lo;
    for (unsigned i = 0; i < static_cast<unsigned>(num_moduli); ++i) {
        gemm_low_prec_i8x1(
            stream,
            handle,
            static_cast<int>(ldc_hi),
            static_cast<int>(n),
            static_cast<int>(lda_lo),
            &one,
            A_lo_tmp,
            lda_lo,
            B_lo_tmp,
            ldb_lo,
            &zero,
            C_hi,
            ldc_hi);
        A_lo_tmp += sizeA;
        B_lo_tmp += sizeB;
        real::conv_hi2mid<gemmul8::Backend::INT8>(
            stream,
            i,
            sizeC_4,
            C_hi,
            C_mid + static_cast<int64_t>(i) * sizeC);
    }

    triple_inverse_split(
        stream,
        static_cast<unsigned>(num_moduli),
        m,
        n,
        C_mid,
        ldc_hi,
        sizeC,
        out_hi.data_ptr<float>(),
        out_mid.data_ptr<float>(),
        out_lo.data_ptr<float>(),
        m,
        sftA,
        sftB,
        has_bias ? bias_hi_c.data_ptr<float>() : nullptr,
        has_bias ? bias_mid_c.data_ptr<float>() : nullptr,
        has_bias ? bias_lo_c.data_ptr<float>() : nullptr,
        has_bias);
    C10_CUDA_CHECK(cudaGetLastError());
    cleanup_handle(handle);
    return {out_hi, out_mid, out_lo};
}

torch::Tensor ozaki_linear_backward_input_cuda(
    torch::Tensor grad_output,
    torch::Tensor weight,
    int64_t num_moduli,
    bool fastmode) {
    return ozaki_matmul_2d_cuda_impl(grad_output, weight, num_moduli, fastmode);
}

torch::Tensor ozaki_linear_backward_weight_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    int64_t num_moduli,
    bool fastmode) {
    return ozaki_matmul_2d_cuda_impl(grad_output.transpose(0, 1), input, num_moduli, fastmode);
}

std::vector<torch::Tensor> triple_sum_dim0_cuda(
    torch::Tensor hi,
    torch::Tensor mid,
    torch::Tensor lo) {
    TORCH_CHECK(hi.is_cuda(), "hi must be a CUDA tensor");
    TORCH_CHECK(hi.dim() == 2, "hi must be 2D");
    check_limb_tensor("mid", mid, hi.device(), hi.sizes());
    check_limb_tensor("lo", lo, hi.device(), hi.sizes());
    TORCH_CHECK(hi.scalar_type() == torch::kFloat32, "hi must be float32");

    const c10::cuda::CUDAGuard device_guard(hi.device());
    const auto rows = hi.size(0);
    const auto cols = hi.size(1);
    auto out_hi = torch::empty({cols}, hi.options());
    auto out_mid = torch::empty({cols}, hi.options());
    auto out_lo = torch::empty({cols}, hi.options());
    if (rows == 0 || cols == 0) {
        out_hi.zero_();
        out_mid.zero_();
        out_lo.zero_();
        return {out_hi, out_mid, out_lo};
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    triple_sum_dim0_kernel<<<static_cast<unsigned>(cols), 256, 0, stream>>>(
        hi.contiguous().data_ptr<float>(),
        mid.contiguous().data_ptr<float>(),
        lo.contiguous().data_ptr<float>(),
        out_hi.data_ptr<float>(),
        out_mid.data_ptr<float>(),
        out_lo.data_ptr<float>(),
        rows,
        cols);
    C10_CUDA_CHECK(cudaGetLastError());
    return {out_hi, out_mid, out_lo};
}

void ozaki_clear_cache() {
    get_work_cache().clear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &ozaki_matmul_cuda, "Ozaki INT8 2D matmul");
    m.def("forward", &ozaki_linear_forward_cuda, "Ozaki INT8 Linear forward");
    m.def("forward_triple", &ozaki_linear_forward_triple_cuda, "Ozaki INT8 Linear forward with float32 triple-limb API");
    m.def("backward_input", &ozaki_linear_backward_input_cuda, "Ozaki INT8 Linear backward input");
    m.def("backward_weight", &ozaki_linear_backward_weight_cuda, "Ozaki INT8 Linear backward weight");
    m.def("sum_dim0_triple", &triple_sum_dim0_cuda, "SGFloat32 sum over dim 0 for triple limbs");
    m.def("clear_cache", &ozaki_clear_cache, "Clear Ozaki temporary work buffer cache");
}
