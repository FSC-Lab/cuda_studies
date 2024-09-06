#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <random>

using BinaryOperation = void (*)(uint, double const*, double const*, double*);

#define MAKE_BINARY_OP(name, op)                                       \
  __device__ void name(uint len, double const* lhs, double const* rhs, \
                       double* res) {                                  \
    for (uint i = 0; i < len; ++i) {                                   \
      res[i] = lhs[i] op rhs[i];                                       \
    }                                                                  \
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
                                      uint len_x, double* lhs, double* rhs,
                                      double* res) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_samples) {
    return;
  }
  double* ith_lhs = lhs + i * len_x;
  double* ith_rhs = rhs + i * len_x;
  double* ith_res = res + i * len_x;
  (*op)(len_x, ith_lhs, ith_rhs, ith_res);
}

thrust::host_vector<double> InvokeBinaryOperation(
    const BinaryOperation& op, const thrust::host_vector<double>& lhs,
    uint num_samples, const thrust::host_vector<double>& rhs) {
  BinaryOperation p_op;
  cudaMemcpyFromSymbol(&p_op, op, sizeof(BinaryOperation));

  thrust::device_vector<double> d_lhs(lhs.cbegin(), lhs.cend());
  thrust::device_vector<double> d_rhs(rhs.cbegin(), rhs.cend());

  thrust::device_vector<double> d_res(lhs.size());
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
  uint num_samples = 100;
  uint len_x = 4;

  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_real_distribution<> dist{10.0, 100.0};
  std::function<double()> generate_fn = std::bind(dist, std::ref(rng));

  thrust::host_vector<double> lhs;
  thrust::host_vector<double> rhs;
  thrust::host_vector<double> res;

  TestFunctionPointer()
      : ::testing::Test(), lhs(num_samples * len_x), rhs(num_samples * len_x) {
    thrust::generate(lhs.begin(), lhs.end(), generate_fn);
    thrust::generate(rhs.begin(), rhs.end(), generate_fn);
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
