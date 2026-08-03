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
#include <string>
#include <unordered_map>
#include <vector>

namespace {

enum class OzakiBackend {
    INT8,
    FP8,
};

OzakiBackend parse_backend(const std::string &backend) {
    if (backend == "int8" || backend == "int8_ozaki2") {
        return OzakiBackend::INT8;
    }
    if (backend == "fp8" || backend == "fp8_ozaki2") {
        return OzakiBackend::FP8;
    }
    if (backend == "mxfp8" || backend == "mxfp8_ozaki2" ||
        backend == "nvfp4" || backend == "nvfp4_ozaki2") {
        TORCH_CHECK(
            false,
            "Ozaki backend '",
            backend,
            "' requires a dedicated C++/CUDA backend; no alternate backend is used");
    }
    TORCH_CHECK(false, "unknown Ozaki backend: ", backend);
    return OzakiBackend::INT8;
}

template <OzakiBackend backend>
struct Gemmul8Backend;

template <>
struct Gemmul8Backend<OzakiBackend::INT8> {
    static constexpr gemmul8::Backend value = gemmul8::Backend::INT8;
};

template <>
struct Gemmul8Backend<OzakiBackend::FP8> {
    static constexpr gemmul8::Backend value = gemmul8::Backend::FP8;
};

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


template <OzakiBackend backend>
torch::Tensor ozaki_matmul_2d_cuda_impl_backend(
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
    constexpr gemmul8::Backend gemmul8_backend = Gemmul8Backend<backend>::value;

    const size_t lwork = gemmul8::workSize<false, gemmul8_backend>(
        m, n, k, static_cast<unsigned>(num_moduli));
    torch::Tensor work = torch::empty(
        {static_cast<int64_t>(lwork)},
        a.options().dtype(torch::kUInt8));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());
    gemmul8::gemmLt<double, gemmul8_backend>(
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

torch::Tensor ozaki_matmul_2d_cuda_impl(
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_moduli,
    bool fastmode,
    const std::string &backend) {
    switch (parse_backend(backend)) {
    case OzakiBackend::INT8:
        return ozaki_matmul_2d_cuda_impl_backend<OzakiBackend::INT8>(
            a, b, num_moduli, fastmode);
    case OzakiBackend::FP8:
        return ozaki_matmul_2d_cuda_impl_backend<OzakiBackend::FP8>(
            a, b, num_moduli, fastmode);
    }
    TORCH_CHECK(false, "unreachable Ozaki backend");
    return torch::Tensor();
}

} // namespace

torch::Tensor ozaki_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_moduli,
    bool fastmode,
    const std::string &backend) {
    return ozaki_matmul_2d_cuda_impl(a, b, num_moduli, fastmode, backend);
}

torch::Tensor ozaki_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t num_moduli,
    bool fastmode,
    const std::string &backend) {
    torch::Tensor output = ozaki_matmul_2d_cuda_impl(
        input, weight.transpose(0, 1), num_moduli, fastmode, backend);
    check_bias(output, bias);
    if (bias.has_value() && bias.value().defined()) {
        output.add_(bias.value().contiguous());
    }
    return output;
}


torch::Tensor ozaki_linear_backward_input_cuda(
    torch::Tensor grad_output,
    torch::Tensor weight,
    int64_t num_moduli,
    bool fastmode,
    const std::string &backend) {
    return ozaki_matmul_2d_cuda_impl(grad_output, weight, num_moduli, fastmode, backend);
}

torch::Tensor ozaki_linear_backward_weight_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    int64_t num_moduli,
    bool fastmode,
    const std::string &backend) {
    return ozaki_matmul_2d_cuda_impl(
        grad_output.transpose(0, 1), input, num_moduli, fastmode, backend);
}


void ozaki_clear_cache() {
    get_work_cache().clear();
}

void ozaki_register_digit_bindings(pybind11::module_ &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &ozaki_matmul_cuda, "Ozaki 2D matmul with selectable INT8/FP8 GEMMul8 backend");
    m.def("forward", &ozaki_linear_forward_cuda, "Ozaki Linear forward with selectable INT8/FP8 GEMMul8 backend");
    m.def("backward_input", &ozaki_linear_backward_input_cuda, "Ozaki Linear backward input with selectable INT8/FP8 GEMMul8 backend");
    m.def("backward_weight", &ozaki_linear_backward_weight_cuda, "Ozaki Linear backward weight with selectable INT8/FP8 GEMMul8 backend");
    m.def("clear_cache", &ozaki_clear_cache, "Clear Ozaki temporary work buffer cache");
    ozaki_register_digit_bindings(m);
}
