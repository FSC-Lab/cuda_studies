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

using BinaryOperation = void (*)(uint, float const*, float const*, float*);

#define MAKE_BINARY_OP(name, op)                                     \
  __device__ void name(uint len, float const* lhs, float const* rhs, \
                       float* res) {                                 \
    for (uint i = 0; i < len; ++i) {                                 \
      res[i] = lhs[i] op rhs[i];                                     \
    }                                                                \
  }

MAKE_BINARY_OP(Add, +);
MAKE_BINARY_OP(Sub, -);
MAKE_BINARY_OP(Mul, *);
MAKE_BINARY_OP(Div, /);
// Required for functional pointer argument in kernel function
// Static pointers to device functions
__device__ BinaryOperation p_add = Add;
__device__ BinaryOperation p_sub = Sub;
__device__ BinaryOperation p_mul = Mul;
__device__ BinaryOperation p_div = Div;

__global__ void BinaryOperationKernel(BinaryOperation op, uint num_samples,
                                      uint len_x, float* lhs, float* rhs,
                                      float* res) {
  ulong i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_samples) {
    return;
  }
  float* ith_lhs = lhs + i * len_x;
  float* ith_rhs = rhs + i * len_x;
  float* ith_res = res + i * len_x;
  (*op)(len_x, ith_lhs, ith_rhs, ith_res);
}

thrust::host_vector<float> InvokeBinaryOperation(
    const BinaryOperation& op, const thrust::host_vector<float>& lhs,
    uint num_samples, const thrust::host_vector<float>& rhs) {
  BinaryOperation p_op;
  cudaMemcpyFromSymbol(&p_op, op, sizeof(BinaryOperation));

  thrust::device_vector<float> d_lhs(lhs.cbegin(), lhs.cend());
  thrust::device_vector<float> d_rhs(rhs.cbegin(), rhs.cend());

  thrust::device_vector<float> d_res(lhs.size());
  const uint threads_per_block = 16;
  const uint num_blocks =
      (num_samples + threads_per_block - 1) / threads_per_block;
  const uint len_x = lhs.size() / num_samples;
  BinaryOperationKernel<<<threads_per_block, num_blocks>>>(
      p_op, num_samples, len_x, d_lhs.data().get(), d_rhs.data().get(),
      d_res.data().get());
  cudaDeviceSynchronize();

  return {d_res.cbegin(), d_res.cend()};
}

struct TestFunctionPointer : public testing::Test {
  ulong num_samples = 100;
  ulong len_x = 4;

  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_real_distribution<float> dist{10.0, 100.0};
  float Generate() { return dist(rng); }

  thrust::host_vector<float> lhs;
  thrust::host_vector<float> rhs;
  thrust::host_vector<float> res;

  TestFunctionPointer() : lhs(num_samples * len_x), rhs(num_samples * len_x) {
    thrust::generate(lhs.begin(), lhs.end(), [this] { return Generate(); });
    thrust::generate(rhs.begin(), rhs.end(), [this] { return Generate(); });
  }
};

TEST_F(TestFunctionPointer, testAddByFunctionPointer) {
  res = InvokeBinaryOperation(p_add, lhs, num_samples, rhs);

  for (int i = 0; i < num_samples * len_x; ++i) {
    ASSERT_NEAR(res[i], lhs[i] + rhs[i], 1e-5);
  }
}

TEST_F(TestFunctionPointer, testSubByFunctionPointer) {
  res = InvokeBinaryOperation(p_sub, lhs, num_samples, rhs);

  for (int i = 0; i < num_samples * len_x; ++i) {
    ASSERT_NEAR(res[i], lhs[i] - rhs[i], 1e-5);
  }
}

TEST_F(TestFunctionPointer, testMulByFunctionPointer) {
  res = InvokeBinaryOperation(p_mul, lhs, num_samples, rhs);

  for (int i = 0; i < num_samples * len_x; ++i) {
    ASSERT_NEAR(res[i], lhs[i] * rhs[i], 1e-5);
  }
}

TEST_F(TestFunctionPointer, testDivByFunctionPointer) {
  res = InvokeBinaryOperation(p_div, lhs, num_samples, rhs);

  for (int i = 0; i < num_samples * len_x; ++i) {
    ASSERT_NEAR(res[i], lhs[i] / rhs[i], 1e-5);
  }
}
