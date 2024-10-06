#include <random>

#include "Eigen/Dense"
#define BOOST_NO_CXX17_HDR_EXECUTION 1
#include "boost/math/statistics/anderson_darling.hpp"
#include "curand_kernel.h"
#include "gtest/gtest.h"
#include "thrust/device_vector.h"

__global__ void matrixGen(float* data, Eigen::Index rows, Eigen::Index cols) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > cols) {
    return;
  }

  curandState state;

  curand_init(tid, 0UL, 0UL, &state);

  Eigen::Map<Eigen::VectorXf> mapped_data{data + tid * rows, rows};

  mapped_data = Eigen::VectorXf::NullaryExpr(
      rows, [p_state{&state}] { return curand_normal(p_state); });
}

namespace ix = Eigen::indexing;
TEST(TestRng, testMatrixGeneration) {
  using boost::math::statistics::anderson_darling_normality_statistic;

  Eigen::MatrixXf data = Eigen::MatrixXf::Zero(8192, 20);

  thrust::device_vector<float> dev_data(data.size());
  thrust::copy_n(data.data(), data.size(), dev_data.begin());
  constexpr auto kBlockSize = 32;
  const auto thread_per_block =
      std::min(1L, (data.size() + kBlockSize - 1) / kBlockSize);

  matrixGen<<<thread_per_block, kBlockSize>>>(dev_data.data().get(),
                                              data.rows(), data.cols());
  thrust::copy(dev_data.cbegin(), dev_data.cend(), data.data());

  for (int i = 0; i < data.cols(); ++i) {
    Eigen::Ref<Eigen::VectorXf> it = data(ix::all, i);
    std::sort(it.begin(), it.end());
    auto a_sq = anderson_darling_normality_statistic(it, 0.0F, 1.0F);

    ASSERT_LT(a_sq / it.size(), 1e-3F) << "Failed on col: " << i;
  }
}
