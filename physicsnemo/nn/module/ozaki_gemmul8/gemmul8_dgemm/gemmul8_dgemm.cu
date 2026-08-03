#include "gemmul8.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#if defined(__NVCC__)
#include <cuComplex.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

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
#include "gemmul8_real.hpp"

#if !defined(GEMMLt_ARGS)
#define GEMMLt_ARGS(T) cublasLtHandle_t handle,                                \
                       cublasOperation_t op_A, cublasOperation_t op_B,         \
                       size_t m, size_t n, size_t k,                           \
                       const T *alpha, const T *const A, size_t lda,           \
                       const T *const B, size_t ldb,                           \
                       const T *beta, T *const C, size_t ldc,                  \
                       unsigned num_moduli, bool fastmode,                     \
                       void *const work, void *const workA, void *const workB, \
                       bool enable_skip_scalA, bool enable_skip_scalB,         \
                       bool skip_scalA, bool skip_scalB,                       \
                       cudaStream_t stream
#endif

#if !defined(GEMMLt_CALL_ARGS)
#define GEMMLt_CALL_ARGS Handle_t(CublasLtTag{}, handle), op_A, op_B, m, n, k, \
                         alpha, A, lda, B, ldb, beta, C, ldc,                  \
                         num_moduli, fastmode, work, workA, workB,             \
                         enable_skip_scalA, enable_skip_scalB,                 \
                         skip_scalA, skip_scalB, stream
#endif

namespace gemmul8 {

template <>
size_t workSize<false, Backend::INT8>(
    size_t m,
    size_t n,
    size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    size_t *workSizeA,
    size_t *workSizeB) {
    return real::workSize<Backend::INT8>(
        m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

template <>
std::vector<double> gemmLt<double, Backend::INT8>(GEMMLt_ARGS(double)) {
    return real::gemm<double, Backend::INT8>(GEMMLt_CALL_ARGS);
}

template <>
size_t workSize<false, Backend::FP8>(
    size_t m,
    size_t n,
    size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    size_t *workSizeA,
    size_t *workSizeB) {
    return real::workSize<Backend::FP8>(
        m, n, k, num_moduli, enable_skip_scalA, enable_skip_scalB, workSizeA, workSizeB);
}

template <>
std::vector<double> gemmLt<double, Backend::FP8>(GEMMLt_ARGS(double)) {
    return real::gemm<double, Backend::FP8>(GEMMLt_CALL_ARGS);
}

} // namespace gemmul8
