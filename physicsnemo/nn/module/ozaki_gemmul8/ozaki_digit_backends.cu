// Ozaki-II digit-decomposition backends with real MXFP8 / NVFP4 hardware GEMMs.
//
// Fused-kernel implementation (v3). Architecture follows the GEMMul8 model:
// one fused extraction pass, grouped (or looped) low-precision GEMMs into
// preallocated buffers, then a single fused reconstruction+CRT kernel —
// instead of dozens of aten-level passes per modulus.
//
// Pipeline (identical algorithm to the Week 9 reference and to v1):
//
//   FP64 inputs
//   -> per-row / per-column power-of-2 scaling shifts (Ozaki-II accurate mode)
//   -> [fused kernel] scale to int64, centered residues per modulus, digit
//      decomposition, and hardware-format packing:
//        MXFP8: radix 16, 2 digits + Karatsuba sum operand, e4m3 bytes
//        NVFP4: beta 9, 3 digits in [-4,4], packed e2m1fn_x2 pairs
//   -> digit GEMMs on real hardware via torch _scaled_mm_v2
//      (float8_e4m3fn / float4_e2m1fn_x2, unit block scales, zero-copy views
//      into the extraction buffers)
//   -> [fused kernel] integer product reconstruction and modular reduction
//   -> [fused kernel] signed CRT reconstruction in double-double arithmetic
//      fused with inverse scaling, FP64 output
//
// All digit values are small integers that are exact in e4m3 / e2m1, and the
// FP32 accumulation of the hardware GEMMs is exact for these ranges, so the
// digit GEMM layer is exact integer arithmetic.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/_scaled_grouped_mm_v2.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int64_t kNvfp4Moduli[] = {727, 719, 709, 701, 691, 683, 677,
                                    673, 661, 659, 653, 647, 643, 641};
constexpr int64_t kMxfp8Moduli[] = {511, 509, 503, 499, 491, 487, 481,
                                    479, 467, 463, 461, 457, 449, 443};
constexpr int64_t kNumModuliMax = 14;

// NVFP4 full 3x3 digit GEMM schedule: {left digit, right digit, beta power}.
constexpr int64_t kNvfp4Schedule[9][3] = {
    {2, 2, 4}, {2, 1, 3}, {2, 0, 2}, {1, 2, 3}, {1, 1, 2},
    {1, 0, 1}, {0, 2, 2}, {0, 1, 1}, {0, 0, 0},
};
constexpr int kNvfp4NumGemms = 9;
constexpr int kMxfp8NumGemms = 3;

// e2m1 4-bit codes for digit values -4..4 (index = digit + 4).
__device__ __constant__ unsigned char kE2m1Lut[9] = {14, 13, 12, 10, 0,
                                                     2,  4,  5,  6};

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t imod64(int64_t a, int64_t m) {
  int64_t r = a % m;
  return r < 0 ? r + m : r;
}

__device__ __forceinline__ int64_t floordiv64(int64_t a, int64_t m) {
  // m > 0
  int64_t q = a / m;
  if (a < 0 && a % m != 0) {
    q -= 1;
  }
  return q;
}

__device__ __forceinline__ int64_t centered_residue_dev(int64_t value,
                                                        int64_t modulus) {
  int64_t r = imod64(value, modulus);
  if (r != 0 && 2 * r >= modulus) {
    r -= modulus;
  }
  return r;
}

__device__ __forceinline__ unsigned char fp8_byte(int digit) {
  return __nv_cvt_float_to_fp8(static_cast<float>(digit), __NV_SATFINITE,
                               __NV_E4M3);
}

// Dekker TwoProd error term for a*b (device).
__device__ __forceinline__ double prod_err(double a, double b) {
  constexpr double SPLITTER = 134217729.0; // 2^27 + 1
  double p = a * b;
  double as = a * SPLITTER;
  double a_hi = as - (as - a);
  double a_lo = a - a_hi;
  double bs = b * SPLITTER;
  double b_hi = bs - (bs - b);
  double b_lo = b - b_hi;
  return ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

// Accumulate r*h into the double-double accumulator (hi, lo).
__device__ __forceinline__ void dd_accumulate(double r, double h, double &hi,
                                              double &lo) {
  double p = r * h;
  double e = prod_err(r, h);
  double s = hi + p;
  double z = s - hi;
  double err = (hi - (s - z)) + (p - z);
  hi = s;
  lo += (err + e);
}

// Finalize one double-double CRT accumulator: subtract the centered multiple
// of the modulus product and return the FP64 result.
__device__ __forceinline__ double crt_finalize(double hi, double lo,
                                               double product_hi,
                                               double product_lo, int shift) {
  const double product_hi_s = ldexp(product_hi, -shift);
  const double product_lo_s = ldexp(product_lo, -shift);
  const double quotient = rint((hi + lo) / (product_hi_s + product_lo_s));
  const double pq = quotient * product_hi_s;
  const double eq = prod_err(quotient, product_hi_s);
  const double s = hi - pq;
  const double z = s - hi;
  const double err = (hi - (s - z)) + (-pq - z);
  const double result_lo = lo + (err - eq) - quotient * product_lo_s;
  return s + result_lo;
}

// ---------------------------------------------------------------------------
// fused extraction kernels
//
// Matrix elements are addressed as val(row, col) = vals[row*stride_row +
// col*stride_col], so the same kernels serve the left matrix (row-major) and
// the right matrix (transposed view). Out-of-range rows/cols write zeros.
// ---------------------------------------------------------------------------

__global__ void mxfp8_extract_kernel(
    const double *__restrict__ vals,
    int64_t stride_row,
    int64_t stride_col,
    const int64_t *__restrict__ shifts,
    int rows_logical,
    int rows_total,
    int cols_logical,
    int cols_padded,
    const int64_t *__restrict__ moduli,
    int num_moduli,
    unsigned char *__restrict__ out) { // [M][3][rows_total][cols_padded]
  const int64_t total = static_cast<int64_t>(rows_total) * cols_padded;
  const size_t plane = static_cast<size_t>(rows_total) * cols_padded;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(idx / cols_padded);
    const int col = static_cast<int>(idx % cols_padded);
    int64_t scaled = 0;
    if (row < rows_logical && col < cols_logical) {
      const double v = vals[static_cast<int64_t>(row) * stride_row +
                            static_cast<int64_t>(col) * stride_col];
      const double mag = ldexp(fabs(v), static_cast<int>(shifts[row]));
      const int64_t t = static_cast<int64_t>(trunc(mag));
      scaled = (v < 0.0) ? -t : t;
    }
    for (int mi = 0; mi < num_moduli; ++mi) {
      const int64_t modulus = moduli[mi];
      const int64_t r = centered_residue_dev(scaled, modulus);
      int d0 = static_cast<int>(imod64(r + 8, 16) - 8);
      int d1 = static_cast<int>(floordiv64(r - d0, 16));
      const int digit_sum = d0 + d1;
      if (digit_sum > 16) {
        d0 -= 16;
        d1 += 1;
      } else if (digit_sum < -16) {
        d0 += 16;
        d1 -= 1;
      }
      const size_t base =
          (static_cast<size_t>(mi) * 3 * rows_total + row) * cols_padded + col;
      out[base] = fp8_byte(d0);
      out[base + plane] = fp8_byte(d1);
      out[base + 2 * plane] = fp8_byte(d0 + d1);
    }
  }
}

__global__ void nvfp4_extract_kernel(
    const double *__restrict__ vals,
    int64_t stride_row,
    int64_t stride_col,
    const int64_t *__restrict__ shifts,
    int rows_logical,
    int rows_total,
    int cols_logical,
    int cols_padded,
    const int64_t *__restrict__ moduli,
    int num_moduli,
    unsigned char *__restrict__ out) { // [M][3][rows_total][cols_padded/2]
  const int64_t pair_cols = cols_padded / 2;
  const int64_t total = static_cast<int64_t>(rows_total) * pair_cols;
  const size_t plane = static_cast<size_t>(rows_total) * pair_cols;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(idx / pair_cols);
    const int pair = static_cast<int>(idx % pair_cols);
    int64_t scaled[2] = {0, 0};
    if (row < rows_logical) {
      for (int t = 0; t < 2; ++t) {
        const int col = 2 * pair + t;
        if (col < cols_logical) {
          const double v = vals[static_cast<int64_t>(row) * stride_row +
                                static_cast<int64_t>(col) * stride_col];
          const double mag = ldexp(fabs(v), static_cast<int>(shifts[row]));
          const int64_t tt = static_cast<int64_t>(trunc(mag));
          scaled[t] = (v < 0.0) ? -tt : tt;
        }
      }
    }
    for (int mi = 0; mi < num_moduli; ++mi) {
      const int64_t modulus = moduli[mi];
      unsigned char bytes[3] = {0, 0, 0};
      for (int t = 0; t < 2; ++t) {
        const int64_t r = centered_residue_dev(scaled[t], modulus);
        const int d0 = static_cast<int>(imod64(r + 4, 9) - 4);
        const int64_t q1 = floordiv64(r - d0, 9);
        const int d1 = static_cast<int>(imod64(q1 + 4, 9) - 4);
        const int d2 = static_cast<int>(floordiv64(q1 - d1, 9));
        const int shift = 4 * t;
        bytes[0] |= kE2m1Lut[d0 + 4] << shift;
        bytes[1] |= kE2m1Lut[d1 + 4] << shift;
        bytes[2] |= kE2m1Lut[d2 + 4] << shift;
      }
      const size_t base =
          (static_cast<size_t>(mi) * 3 * rows_total + row) * pair_cols + pair;
      out[base] = bytes[0];
      out[base + plane] = bytes[1];
      out[base + 2 * plane] = bytes[2];
    }
  }
}

// ---------------------------------------------------------------------------
// fused reconstruction + signed CRT + inverse scaling
//
// One thread per output element: loops over the moduli, reconstructs the
// integer product from the digit-GEMM outputs in registers, reduces it
// modulo each modulus, and accumulates the double-double CRT — no
// residue_terms round trip.
// ---------------------------------------------------------------------------

__global__ void mxfp8_fused_crt_kernel(
    const float *__restrict__ gemm_out_all, // [3][M][m][n_al]: hh, ll, ss
    int num_moduli,
    int m,
    int n,
    int n_al,
    const int64_t *__restrict__ moduli,
    const double *__restrict__ basis_hi,
    const double *__restrict__ basis_lo,
    double product_hi,
    double product_lo,
    const int64_t *__restrict__ left_shifts,
    const int64_t *__restrict__ right_shifts,
    double *__restrict__ out) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<int64_t>(m) * n) {
    return;
  }
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  const int shift = static_cast<int>(left_shifts[i] + right_shifts[j]);
  const size_t plane = static_cast<size_t>(m) * n_al;
  const size_t off = static_cast<size_t>(i) * n_al + j;

  double hi = 0.0;
  double lo = 0.0;
  for (int mi = 0; mi < num_moduli; ++mi) {
    const double hi_hi = gemm_out_all[static_cast<size_t>(mi) * plane + off];
    const double lo_lo =
        gemm_out_all[(static_cast<size_t>(num_moduli) + mi) * plane + off];
    const double sum_sum =
        gemm_out_all[(2 * static_cast<size_t>(num_moduli) + mi) * plane + off];
    const double cross = sum_sum - hi_hi - lo_lo;
    const double rec = 256.0 * hi_hi + 16.0 * cross + lo_lo;
    const double modulus = static_cast<double>(moduli[mi]);
    double r = remainder(round(rec), modulus);
    if (r < 0.0) {
      r += modulus;
    }
    dd_accumulate(r, ldexp(basis_hi[mi], -shift), hi, lo);
    dd_accumulate(r, ldexp(basis_lo[mi], -shift), hi, lo);
  }
  out[idx] = crt_finalize(hi, lo, product_hi, product_lo, shift);
}

__global__ void nvfp4_fused_crt_kernel(
    const float *__restrict__ gemm_out_all, // [9][M][m][n_al] in schedule order
    int num_moduli,
    int m,
    int n,
    int n_al,
    const int64_t *__restrict__ moduli,
    const double *__restrict__ basis_hi,
    const double *__restrict__ basis_lo,
    double product_hi,
    double product_lo,
    const int64_t *__restrict__ left_shifts,
    const int64_t *__restrict__ right_shifts,
    double *__restrict__ out) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<int64_t>(m) * n) {
    return;
  }
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  const int shift = static_cast<int>(left_shifts[i] + right_shifts[j]);
  const size_t plane = static_cast<size_t>(m) * n_al;
  const size_t off = static_cast<size_t>(i) * n_al + j;
  // 9^beta_power of kNvfp4Schedule: beta powers are 4,3,2,3,2,1,2,1,0.
  constexpr int powers[kNvfp4NumGemms] = {6561, 729, 81, 729, 81,
                                          9,    81,  9,  1};

  double hi = 0.0;
  double lo = 0.0;
  for (int mi = 0; mi < num_moduli; ++mi) {
    double rec = 0.0;
    for (int g = 0; g < kNvfp4NumGemms; ++g) {
      rec += static_cast<double>(powers[g]) *
             gemm_out_all[(static_cast<size_t>(g) * num_moduli + mi) * plane +
                          off];
    }
    const double modulus = static_cast<double>(moduli[mi]);
    double r = remainder(round(rec), modulus);
    if (r < 0.0) {
      r += modulus;
    }
    dd_accumulate(r, ldexp(basis_hi[mi], -shift), hi, lo);
    dd_accumulate(r, ldexp(basis_lo[mi], -shift), hi, lo);
  }
  out[idx] = crt_finalize(hi, lo, product_hi, product_lo, shift);
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------

std::vector<int64_t> moduli_for(const std::string &backend, int64_t num_moduli) {
  const int64_t *table = (backend == "nvfp4") ? kNvfp4Moduli : kMxfp8Moduli;
  const int64_t count = std::min<int64_t>(num_moduli, kNumModuliMax);
  TORCH_CHECK(count > 0, "num_moduli must be positive");
  return std::vector<int64_t>(table, table + count);
}

double log2p_for(const std::vector<int64_t> &moduli) {
  double sum = 0.0;
  for (int64_t m : moduli) {
    sum += std::log2(static_cast<double>(m));
  }
  return std::floor(sum / 2 - 0.5);
}

// Ozaki-II accurate-mode per-row (or per-column) scaling shift:
//   shift = -floor(log2(amax)) + floor(-0.5*log2(k) + log2P) - 1
// plus an int64 safety cap so scaled values never overflow torch int64.
torch::Tensor safe_shifts(const torch::Tensor &amax, int64_t k, double log2p) {
  const int64_t shift_base = static_cast<int64_t>(
      std::floor(-0.5 * std::log2(static_cast<double>(k)) + log2p)) - 1;
  auto positive = amax > 0;
  auto safe_amax = torch::where(positive, amax, torch::ones_like(amax));
  auto log2_amax = torch::floor(torch::log2(safe_amax)).to(torch::kLong);
  auto shifts = -log2_amax + shift_base;
  shifts = torch::minimum(shifts, 61 - log2_amax);
  return torch::where(positive, shifts, torch::zeros_like(shifts));
}

// Flat all-ones block-scale tensor with the blocked-layout element count.
// The scales are uniformly 1.0, so the swizzle permutation is immaterial;
// only the padded element count must match what _scaled_mm_v2 expects.
torch::Tensor blocked_ones(int64_t rows, int64_t blocks,
                           torch::ScalarType dtype,
                           const torch::TensorOptions &options) {
  const int64_t row_blocks = (rows + 127) / 128;
  const int64_t col_blocks = (blocks + 3) / 4;
  const int64_t numel = row_blocks * 128 * col_blocks * 4;
  if (dtype == torch::kFloat8_e8m0fnu) {
    // e8m0 encoding of 1.0 is exponent bias 127 = 0x7F.
    auto bytes = torch::full({numel}, 127, options.dtype(torch::kUInt8));
    return bytes.view(dtype);
  }
  return torch::ones({numel}, options).to(dtype);
}

torch::Tensor device_i64(const std::vector<int64_t> &values,
                         const torch::Device &device) {
  return torch::from_blob(const_cast<int64_t *>(values.data()),
                          {static_cast<int64_t>(values.size())},
                          torch::TensorOptions().dtype(torch::kLong))
      .to(device);
}

torch::Tensor device_f64(const std::vector<double> &values,
                         const torch::Device &device) {
  return torch::from_blob(const_cast<double *>(values.data()),
                          {static_cast<int64_t>(values.size())},
                          torch::TensorOptions().dtype(torch::kDouble))
      .to(device);
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + 255) / 256;
  return static_cast<int>(std::min<int64_t>(blocks, 65535 * 8));
}

} // namespace

struct DigitPipelineResult {
  torch::Tensor output;
  torch::Tensor left_shifts;
  torch::Tensor right_shifts;
  torch::Tensor left_buf;
  torch::Tensor right_buf;
  torch::Tensor gemm_out_all;
};

DigitPipelineResult run_digit_pipeline(
    torch::Tensor &a,
    torch::Tensor &b,
    int64_t num_moduli,
    const std::string &backend,
    std::vector<double> &crt_basis_hi,
    std::vector<double> &crt_basis_lo,
    double crt_product_hi,
    double crt_product_lo) {
  TORCH_CHECK(backend == "mxfp8" || backend == "nvfp4",
              "digit matmul backend must be 'mxfp8' or 'nvfp4', got ", backend);
  TORCH_CHECK(a.scalar_type() == torch::kFloat64 &&
                  b.scalar_type() == torch::kFloat64,
              "digit matmul requires float64 inputs");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "digit matmul requires CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "digit matmul supports 2D only");
  TORCH_CHECK(a.size(1) == b.size(0), "a and b inner dimensions must match");

  const c10::cuda::CUDAGuard device_guard(a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  const int64_t m = a_c.size(0);
  const int64_t k = a_c.size(1);
  const int64_t n = b_c.size(1);
  auto output = torch::empty({m, n}, a_c.options());
  if (m == 0 || n == 0) {
    return {output, output, output, output, output, output};
  }
  if (k == 0) {
    output.zero_();
    return {output, output, output, output, output, output};
  }

  const auto moduli = moduli_for(backend, num_moduli);
  const int64_t moduli_count = static_cast<int64_t>(moduli.size());
  TORCH_CHECK(static_cast<int64_t>(crt_basis_hi.size()) == moduli_count &&
                  static_cast<int64_t>(crt_basis_lo.size()) == moduli_count,
              "crt_basis sizes must match the number of moduli");
  const double log2p = log2p_for(moduli);
  const bool is_nvfp4 = (backend == "nvfp4");
  const int64_t gemms_per_modulus = is_nvfp4 ? kNvfp4NumGemms : kMxfp8NumGemms;

  const int64_t k_al = (k + 31) / 32 * 32;
  const int64_t n_al = (n + 15) / 16 * 16;
  const int64_t k_packed = is_nvfp4 ? k_al / 2 : k_al;

  auto left_shifts = safe_shifts(a_c.abs().amax(1), k, log2p);
  auto right_shifts = safe_shifts(b_c.abs().amax(0), k, log2p);

  const auto u8 = a_c.options().dtype(torch::kUInt8);
  auto left_buf = torch::empty({moduli_count, 3, m, k_packed}, u8);
  auto right_buf = torch::empty({moduli_count, 3, n_al, k_packed}, u8);
  auto gemm_out_all = torch::empty(
      {gemms_per_modulus, moduli_count, m, n_al},
      a_c.options().dtype(torch::kFloat));

  auto moduli_dev = device_i64(moduli, a_c.device());
  auto basis_hi_dev = device_f64(crt_basis_hi, a_c.device());
  auto basis_lo_dev = device_f64(crt_basis_lo, a_c.device());

  // Fused extraction: scaling, residues, digit decomposition, HW packing.
  {
    const int threads = 256;
    const int64_t left_total =
        is_nvfp4 ? m * (k_al / 2) : m * k_al;
    const int64_t right_total =
        is_nvfp4 ? n_al * (k_al / 2) : n_al * k_al;
    if (is_nvfp4) {
      nvfp4_extract_kernel<<<blocks_for(left_total), threads, 0, stream>>>(
          a_c.data_ptr<double>(), k, 1, left_shifts.data_ptr<int64_t>(),
          static_cast<int>(m), static_cast<int>(m), static_cast<int>(k),
          static_cast<int>(k_al), moduli_dev.data_ptr<int64_t>(),
          static_cast<int>(moduli_count), left_buf.data_ptr<unsigned char>());
      nvfp4_extract_kernel<<<blocks_for(right_total), threads, 0, stream>>>(
          b_c.data_ptr<double>(), 1, n, right_shifts.data_ptr<int64_t>(),
          static_cast<int>(n), static_cast<int>(n_al), static_cast<int>(k),
          static_cast<int>(k_al), moduli_dev.data_ptr<int64_t>(),
          static_cast<int>(moduli_count), right_buf.data_ptr<unsigned char>());
    } else {
      mxfp8_extract_kernel<<<blocks_for(left_total), threads, 0, stream>>>(
          a_c.data_ptr<double>(), k, 1, left_shifts.data_ptr<int64_t>(),
          static_cast<int>(m), static_cast<int>(m), static_cast<int>(k),
          static_cast<int>(k_al), moduli_dev.data_ptr<int64_t>(),
          static_cast<int>(moduli_count), left_buf.data_ptr<unsigned char>());
      mxfp8_extract_kernel<<<blocks_for(right_total), threads, 0, stream>>>(
          b_c.data_ptr<double>(), 1, n, right_shifts.data_ptr<int64_t>(),
          static_cast<int>(n), static_cast<int>(n_al), static_cast<int>(k),
          static_cast<int>(k_al), moduli_dev.data_ptr<int64_t>(),
          static_cast<int>(moduli_count), right_buf.data_ptr<unsigned char>());
    }
    C10_CUDA_CHECK(cudaGetLastError());
  }

  // Block scales are created once per call (uniform 1.0).
  torch::Tensor scale_a, scale_b, global_a, global_b;
  if (is_nvfp4) {
    scale_a = blocked_ones(m, k_al / 16, torch::kFloat8_e4m3fn, a_c.options());
    scale_b =
        blocked_ones(n_al, k_al / 16, torch::kFloat8_e4m3fn, a_c.options());
    global_a = torch::ones({1}, a_c.options().dtype(torch::kFloat));
    global_b = torch::ones({1}, a_c.options().dtype(torch::kFloat));
  } else {
    scale_a = blocked_ones(m, k_al / 32, torch::kFloat8_e8m0fnu, a_c.options());
    scale_b =
        blocked_ones(n_al, k_al / 32, torch::kFloat8_e8m0fnu, a_c.options());
  }

  // Digit GEMMs on real hardware. Two paths:
  //  - grouped: one _scaled_grouped_mm_v2 call per schedule entry with all
  //    moduli batched (when the runtime supports it);
  //  - loop: one _scaled_mm_v2_out call per (modulus, schedule entry).
  static const bool grouped_disabled =
      std::getenv("OZAKI_DIGIT_NO_GROUPED") != nullptr;
  static int grouped_supported[2] = {-1, -1}; // per backend: -1 unknown
  const int backend_idx = is_nvfp4 ? 1 : 0;

  auto grouped_gemms = [&]() {
    for (int64_t g = 0; g < gemms_per_modulus; ++g) {
      int64_t left_op, right_op;
      if (is_nvfp4) {
        left_op = kNvfp4Schedule[g][0];
        right_op = kNvfp4Schedule[g][1];
      } else {
        left_op = (g == 0) ? 1 : (g == 1 ? 0 : 2);
        right_op = left_op;
      }
      torch::Tensor out_g;
      if (is_nvfp4) {
        auto mat1 = left_buf.select(1, left_op).view(torch::kFloat4_e2m1fn_x2);
        auto mat2 = right_buf.select(1, right_op)
                        .view(torch::kFloat4_e2m1fn_x2)
                        .transpose(1, 2);
        out_g = at::_scaled_grouped_mm_v2(
            mat1, mat2, {scale_a, global_a}, {2, 0}, {1, 0},
            {scale_b, global_b}, {2, 0}, {1, 0}, std::nullopt, std::nullopt,
            torch::kFloat, {}, false);
      } else {
        auto mat1 = left_buf.select(1, left_op).view(torch::kFloat8_e4m3fn);
        auto mat2 = right_buf.select(1, right_op)
                        .view(torch::kFloat8_e4m3fn)
                        .transpose(1, 2);
        out_g = at::_scaled_grouped_mm_v2(
            mat1, mat2, {scale_a}, {3}, {1}, {scale_b}, {3}, {1}, std::nullopt,
            std::nullopt, torch::kFloat, {}, false);
      }
      gemm_out_all[g].copy_(out_g);
    }
  };

  // Loop path: per-(modulus, schedule-entry) _scaled_mm_v2_out calls. The
  // digit GEMMs are mutually independent across moduli, so they are issued
  // across OZAKI_DIGIT_NUM_STREAMS (default 4) side streams and joined back
  // onto the main stream with events. Numerics are identical to the serial
  // version; only the execution schedule changes.
  auto loop_gemms = [&]() {
    int num_streams = 4;
    if (const char *env = std::getenv("OZAKI_DIGIT_NUM_STREAMS")) {
      num_streams = std::max(1, std::atoi(env));
    }
    num_streams =
        static_cast<int>(std::min<int64_t>(num_streams, moduli_count));

    if (num_streams <= 1) {
      for (int64_t mi = 0; mi < moduli_count; ++mi) {
        for (int64_t g = 0; g < gemms_per_modulus; ++g) {
          int64_t left_op, right_op;
          if (is_nvfp4) {
            left_op = kNvfp4Schedule[g][0];
            right_op = kNvfp4Schedule[g][1];
          } else {
            left_op = (g == 0) ? 1 : (g == 1 ? 0 : 2);
            right_op = left_op;
          }
          torch::Tensor mat1, mat2;
          auto out_view = gemm_out_all[g][mi];
          if (is_nvfp4) {
            mat1 = left_buf[mi][left_op].view(torch::kFloat4_e2m1fn_x2);
            mat2 = right_buf[mi][right_op]
                       .view(torch::kFloat4_e2m1fn_x2)
                       .transpose(0, 1);
            at::_scaled_mm_v2_out(out_view, mat1, mat2,
                                  {scale_a, global_a}, {2, 0}, {1, 0},
                                  {scale_b, global_b}, {2, 0}, {1, 0},
                                  std::nullopt, torch::kFloat, {}, false);
          } else {
            mat1 = left_buf[mi][left_op].view(torch::kFloat8_e4m3fn);
            mat2 = right_buf[mi][right_op]
                       .view(torch::kFloat8_e4m3fn)
                       .transpose(0, 1);
            at::_scaled_mm_v2_out(out_view, mat1, mat2, {scale_a}, {3}, {1},
                                  {scale_b}, {3}, {1}, std::nullopt,
                                  torch::kFloat, {}, false);
          }
        }
      }
      return;
    }

    // Multi-stream schedule.
    const auto main_stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAEvent ready;
    ready.record(main_stream);

    std::vector<at::cuda::CUDAStream> streams;
    std::vector<at::cuda::CUDAEvent> done(static_cast<size_t>(num_streams));
    streams.reserve(static_cast<size_t>(num_streams));
    for (int s = 0; s < num_streams; ++s) {
      streams.push_back(
          at::cuda::getStreamFromPool(false, a.device().index()));
      ready.block(streams.back());
    }

    for (int64_t mi = 0; mi < moduli_count; ++mi) {
      const int s = static_cast<int>(mi % num_streams);
      const c10::cuda::CUDAStreamGuard guard(streams[s]);
      for (int64_t g = 0; g < gemms_per_modulus; ++g) {
        int64_t left_op, right_op;
        if (is_nvfp4) {
          left_op = kNvfp4Schedule[g][0];
          right_op = kNvfp4Schedule[g][1];
        } else {
          left_op = (g == 0) ? 1 : (g == 1 ? 0 : 2);
          right_op = left_op;
        }
        torch::Tensor mat1, mat2;
        auto out_view = gemm_out_all[g][mi];
        if (is_nvfp4) {
          mat1 = left_buf[mi][left_op].view(torch::kFloat4_e2m1fn_x2);
          mat2 = right_buf[mi][right_op]
                     .view(torch::kFloat4_e2m1fn_x2)
                     .transpose(0, 1);
          at::_scaled_mm_v2_out(out_view, mat1, mat2,
                                {scale_a, global_a}, {2, 0}, {1, 0},
                                {scale_b, global_b}, {2, 0}, {1, 0},
                                std::nullopt, torch::kFloat, {}, false);
        } else {
          mat1 = left_buf[mi][left_op].view(torch::kFloat8_e4m3fn);
          mat2 = right_buf[mi][right_op]
                     .view(torch::kFloat8_e4m3fn)
                     .transpose(0, 1);
          at::_scaled_mm_v2_out(out_view, mat1, mat2, {scale_a}, {3}, {1},
                                {scale_b}, {3}, {1}, std::nullopt,
                                torch::kFloat, {}, false);
        }
      }
    }

    for (int s = 0; s < num_streams; ++s) {
      done[static_cast<size_t>(s)].record(streams[s]);
      done[static_cast<size_t>(s)].block(main_stream);
    }
  };

  if (!grouped_disabled && grouped_supported[backend_idx] != 0) {
    try {
      grouped_gemms();
      grouped_supported[backend_idx] = 1;
    } catch (const std::exception &) {
      grouped_supported[backend_idx] = 0;
      loop_gemms();
    }
  } else {
    loop_gemms();
  }
  C10_CUDA_CHECK(cudaGetLastError());

  // Fused reconstruction + signed CRT + inverse scaling (single launch).
  if (is_nvfp4) {
    nvfp4_fused_crt_kernel<<<blocks_for(m * n), 256, 0, stream>>>(
        gemm_out_all.data_ptr<float>(), static_cast<int>(moduli_count),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(n_al),
        moduli_dev.data_ptr<int64_t>(), basis_hi_dev.data_ptr<double>(),
        basis_lo_dev.data_ptr<double>(), crt_product_hi, crt_product_lo,
        left_shifts.data_ptr<int64_t>(), right_shifts.data_ptr<int64_t>(),
        output.data_ptr<double>());
  } else {
    mxfp8_fused_crt_kernel<<<blocks_for(m * n), 256, 0, stream>>>(
        gemm_out_all.data_ptr<float>(), static_cast<int>(moduli_count),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(n_al),
        moduli_dev.data_ptr<int64_t>(), basis_hi_dev.data_ptr<double>(),
        basis_lo_dev.data_ptr<double>(), crt_product_hi, crt_product_lo,
        left_shifts.data_ptr<int64_t>(), right_shifts.data_ptr<int64_t>(),
        output.data_ptr<double>());
  }
  C10_CUDA_CHECK(cudaGetLastError());

  return {output, left_shifts, right_shifts, left_buf, right_buf, gemm_out_all};
}

torch::Tensor ozaki_digit_matmul_cuda(
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_moduli,
    const std::string &backend,
    std::vector<double> crt_basis_hi,
    std::vector<double> crt_basis_lo,
    double crt_product_hi,
    double crt_product_lo) {
  auto result = run_digit_pipeline(a, b, num_moduli, backend, crt_basis_hi,
                                   crt_basis_lo, crt_product_hi,
                                   crt_product_lo);
  return result.output;
}

std::vector<torch::Tensor> ozaki_digit_debug_cuda(
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_moduli,
    const std::string &backend,
    std::vector<double> crt_basis_hi,
    std::vector<double> crt_basis_lo,
    double crt_product_hi,
    double crt_product_lo) {
  auto result = run_digit_pipeline(a, b, num_moduli, backend, crt_basis_hi,
                                   crt_basis_lo, crt_product_hi,
                                   crt_product_lo);
  return {result.output,   result.left_shifts, result.right_shifts,
          result.left_buf, result.right_buf,   result.gemm_out_all};
}

void ozaki_register_digit_bindings(pybind11::module_ &m) {
  m.def("matmul_digit", &ozaki_digit_matmul_cuda,
        "Ozaki-II digit-decomposition matmul with real MXFP8/NVFP4 hardware GEMMs (fused kernels)");
  m.def("debug_digit", &ozaki_digit_debug_cuda,
        "Debug: return output plus intermediate pipeline tensors");
}
