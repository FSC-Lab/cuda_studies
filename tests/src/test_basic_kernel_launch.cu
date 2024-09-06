// Copyright © 2024 ADR Team
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <random>

#include "gtest/gtest.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

// Kernel - Adding two matrices MatA and MatB

__global__ void MatAdd(double const* lhs, dim3 size, double const* rhs,
                       double* res) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  auto rows = size.x;
  auto cols = size.y;
  if (i < rows && j < cols) {
    res[i * cols + j] = lhs[i * cols + j] + rhs[i * cols + j];
  }
}

struct TestBasicKernelLaunch : public testing::Test {
  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_real_distribution<> dist{-10.0, 10.0};
  double Generate() { return dist(rng); };
};

TEST_F(TestBasicKernelLaunch, testBasicKernelLaunch) {
  const ulong rows = 100;
  const ulong cols = 100;
  // Matrix addition kernel launch from host code
  const dim3 threads_per_block(16, 16);

  double* d_lhs;
  ASSERT_TRUE(cudaMalloc(&d_lhs, rows * cols * sizeof d_lhs) == cudaSuccess);

  double* d_rhs;
  ASSERT_TRUE(cudaMalloc(&d_rhs, rows * cols * sizeof d_lhs) == cudaSuccess);

  auto* h_lhs = new double[rows * cols];
  auto* h_rhs = new double[rows * cols];
  for (int i = 0; i < rows * cols; ++i) {
    h_lhs[i] = Generate();
    h_rhs[i] = Generate();
  }
  ASSERT_TRUE(cudaMemcpy(d_lhs, h_lhs, rows * cols * sizeof d_lhs,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d_rhs, h_rhs, rows * cols * sizeof d_rhs,
                         cudaMemcpyHostToDevice) == cudaSuccess);

  dim3 num_blocks{static_cast<unsigned int>((rows + threads_per_block.x - 1) /
                                            threads_per_block.x),
                  static_cast<unsigned int>((cols + threads_per_block.y - 1) /
                                            threads_per_block.y)};

  double* d_res;
  ASSERT_TRUE(cudaMalloc(&d_res, rows * cols * sizeof d_res) == cudaSuccess);
  MatAdd<<<num_blocks, threads_per_block>>>(d_lhs, dim3{rows, cols}, d_rhs,
                                            d_res);
  auto* h_res = new double[rows * cols];
  ASSERT_TRUE(cudaMemcpy(h_res, d_res, rows * cols * sizeof h_res,
                         cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < rows * cols; ++i) {
    ASSERT_NEAR(h_res[i], h_lhs[i] + h_rhs[i], 1e-5);
  }

  ASSERT_TRUE(cudaFree(d_lhs) == cudaSuccess);
  ASSERT_TRUE(cudaFree(d_rhs) == cudaSuccess);
  ASSERT_TRUE(cudaFree(d_res) == cudaSuccess);
  delete[] h_lhs;
  delete[] h_rhs;
  delete[] h_res;
}

TEST_F(TestBasicKernelLaunch, testBasicKernelLaunchWithThrust) {
  const ulong rows = 100;
  const ulong cols = 100;
  // Matrix addition kernel launch from host code
  const dim3 threads_per_block(16, 16);

  thrust::host_vector<double> h_lhs(rows * cols);
  thrust::generate(h_lhs.begin(), h_lhs.end(), [this] { return Generate(); });
  thrust::device_vector<double> d_lhs(h_lhs.cbegin(), h_lhs.cend());

  thrust::host_vector<double> h_rhs(rows * cols);
  thrust::generate(h_rhs.begin(), h_rhs.end(), [this] { return Generate(); });
  thrust::device_vector<double> d_rhs(h_rhs.cbegin(), h_rhs.cend());

  dim3 num_blocks{static_cast<unsigned int>((rows + threads_per_block.x - 1) /
                                            threads_per_block.x),
                  static_cast<unsigned int>((cols + threads_per_block.y - 1) /
                                            threads_per_block.y)};

  thrust::device_vector<double> d_res(rows * cols);
  MatAdd<<<num_blocks, threads_per_block>>>(
      d_lhs.data().get(), dim3{rows, cols}, d_rhs.data().get(),
      d_res.data().get());
  thrust::host_vector<double> h_res(d_res.begin(), d_res.end());

  for (int i = 0; i < rows * cols; ++i) {
    ASSERT_NEAR(h_res[i], h_lhs[i] + h_rhs[i], 1e-5);
  }
}
