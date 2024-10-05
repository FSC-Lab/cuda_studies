

#include <cstdio>
#include <random>

#include "Eigen/Dense"
#include "cooperative_groups/reduce.h"
#include "gtest/gtest.h"
#include "thrust/device_vector.h"

namespace cg = cooperative_groups;

namespace details {
template <typename T>
__device__ T thread_sum(T const* input, size_t len);

template <int TileSz = -1, typename T>
__device__ T reduce_sum(cg::thread_block_tile<TileSz> g, void* t, T val) {
  if constexpr (TileSz < 0) {
    assert(t != nullptr);
    auto temp = static_cast<T*>(t);

    auto lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (auto i = g.size() / 2; i > 0; i /= 2) {
      temp[lane] = val;
      g.sync();  // wait for all threads to store
      if (lane < i) {
        val += temp[lane + i];
      }
      g.sync();  // wait for all threads to load
    }
  } else {
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
      val += g.shfl_down(val, i);
    }
  }

  return val;  // note: only thread 0 will return full sum
}

template <typename T>
__device__ T thread_sum(T const* input, size_t len) {
  T sum{0};

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < len / 4U;
       i += blockDim.x * gridDim.x) {
    // The following reinterpretation is safe and blessed by nvidia
    // https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

    if constexpr (std::is_same_v<T, int>) {
      auto in = reinterpret_cast<int4 const*>(input)[i];
      sum += in.x + in.y + in.z + in.w;
    } else if constexpr (std::is_same_v<T, float>) {
      auto in = reinterpret_cast<float4 const*>(input)[i];
      sum += in.x + in.y + in.z + in.w;
    } else {
      sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
    }
  }
  return sum;
}
}  // namespace details

template <typename T>
__global__ void sum_kernel(T* sum, T const* input, size_t n) {
  const auto my_sum = thread_sum(input, n);

  extern __shared__ T temp[];
  cg::thread_group g = cg::this_thread_block();
  const auto block_sum = reduce_sum(g, temp, my_sum);

  if (g.thread_rank() == 0) {
    atomicAdd(sum, block_sum);
  }
}

template <int TileSz, typename T>
__global__ void sum_kernel(T* sum, T const* input, size_t n) {
  const auto my_sum = details::thread_sum(input, n);

  auto tile = cg::tiled_partition<TileSz>(cg::this_thread_block());
  const auto tile_sum = details::reduce_sum<TileSz>(tile, nullptr, my_sum);

  if (tile.thread_rank() == 0) {
    atomicAdd(sum, tile_sum);
  }
}

template <typename It, typename T>
T cuda_accumulate(It begin, It end, T init) {
  const auto len = std::distance(begin, end);

  thrust::device_vector<T> data(len);
  thrust::copy(begin, end, data.begin());

  thrust::device_vector<T> result = {init};

  constexpr int kBlockSize = 256;
  const auto threads_per_block = (len + kBlockSize - 1) / kBlockSize;
  const auto shared_bytes = kBlockSize * sizeof threads_per_block;
  sum_kernel<16><<<threads_per_block, kBlockSize, shared_bytes>>>(
      result.data().get(), data.data().get(), len);
  T scalar_result;
  thrust::copy_n(result.begin(), 1, &scalar_result);
  return scalar_result;
}

struct TestReduction : public testing::TestWithParam<int> {
  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_real_distribution<float> dist{-1.0F, 1.0F};
};

TEST_P(TestReduction, testSum) {
  auto len = GetParam();
  Eigen::VectorXf x =
      Eigen::VectorXf::NullaryExpr(len, [&] { return dist(rng); });
  ASSERT_NEAR(cuda_accumulate(x.begin(), x.end(), 0.0F), x.sum(), len * 1e-8);
}

template <typename T, typename... Ts,
          typename std::enable_if_t<(sizeof...(Ts) > 0)>* = nullptr>
__device__ constexpr T min(const T& x, const T& y, Ts&&... rem) {
  return min(min(x, y), static_cast<Ts&&>(rem)...);
}

__device__ float AtomicMin(float* address, float val) {
  auto* address_as_i = reinterpret_cast<int*>(address);
  auto old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

template <typename T>
__device__ T reduce_min(cooperative_groups::thread_group g, T* temp, T val) {
  auto lane = g.thread_rank();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (auto i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = val;
    g.sync();  // wait for all threads to store
    if (lane < i) {
      val = min(val, temp[lane + i]);
    }
    g.sync();  // wait for all threads to load
  }
  return val;  // note: only thread 0 will return full sum
}

template <typename T>
__device__ T thread_min(T const* input, size_t len) {
  T sum{input[0]};

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < len / 4U;
       i += blockDim.x * gridDim.x) {
    // The following reTerpretation is safe and blessed by nvidia
    // https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

    if constexpr (std::is_same_v<T, int>) {
      auto in = reinterpret_cast<int4 const*>(input)[i];
      sum = min(sum, in.x, in.y, in.z, in.w);
    } else if constexpr (std::is_same_v<T, float>) {
      auto in = reinterpret_cast<float4 const*>(input)[i];
      sum = min(sum, in.x, in.y, in.z, in.w);
    } else {
      sum = min(sum, input[i], input[i + 1], input[i + 2], input[i + 3]);
    }
  }
  return sum;
}

template <typename T>
__global__ void min_kernel_block(T* sum, T const* input, size_t n) {
  const auto my_sum = thread_min(input, n);

  extern __shared__ T temp[];
  cooperative_groups::thread_group g = cooperative_groups::this_thread_block();
  const auto block_sum = reduce_min(g, temp, my_sum);

  if (g.thread_rank() == 0) {
    AtomicMin(sum, block_sum);
  }
}

template <typename It,
          typename T = typename std::iterator_traits<It>::value_type>
T cuda_min(It begin, It end) {
  const auto len = std::distance(begin, end);

  thrust::device_vector<T> data(len);
  thrust::copy(begin, end, data.begin());

  thrust::device_vector<T> result = {std::numeric_limits<T>::max()};

  constexpr int kBlockSize = 256;
  const auto threads_per_block = (len + kBlockSize - 1) / kBlockSize;
  const auto shared_bytes = kBlockSize * sizeof threads_per_block;
  min_kernel_block<<<threads_per_block, kBlockSize, shared_bytes>>>(
      result.data().get(), data.data().get(), len);
  T scalar_result;
  thrust::copy_n(result.begin(), 1, &scalar_result);
  return scalar_result;
}

TEST_P(TestReduction, testMin) {
  auto len = GetParam();
  Eigen::VectorXf x =
      Eigen::VectorXf::NullaryExpr(len, [&] { return dist(rng); });
  ASSERT_NEAR(cuda_min(x.begin(), x.end()), x.minCoeff(), len * 1e-8);
}

INSTANTIATE_TEST_SUITE_P(ins, TestReduction,
                         testing::Values(1 << 10, 1 << 12, 1 << 14, 1 << 16,
                                         1 << 18));
