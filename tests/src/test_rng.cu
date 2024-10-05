#include "Eigen/Dense"
#include "thrust/device_vector.h"
// #include "curand_kernel.h"

__global__ void matrixGen(float *data, Eigen::Index rows, Eigen::Index cols) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > cols) {
    return;
  }

  Eigen::Map<Eigen::VectorXf> mapped_data{data, rows};

  mapped_data = Eigen::VectorXf::NullaryExpr([i] { return i; });
}

int main() {
  Eigen::MatrixXf data = Eigen::MatrixXf::Zero(15, 30);

  thrust::device_vector<float> dev_data(data.size());
  thrust::copy_n(data.data(), data.size(), dev_data.begin());
  constexpr auto kBlockSize = 32;
  const auto thread_per_block =
      std::min(1L, (data.size() + kBlockSize - 1) / kBlockSize);

  matrixGen<<<thread_per_block, kBlockSize>>>(dev_data.data().get(),
                                              data.rows(), data.cols());
  thrust::copy(dev_data.cbegin(), dev_data.cend(), data.data());
}
