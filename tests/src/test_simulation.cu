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

#include "cuda_studies/cpu/ensemble_simulation.hpp"
#include "cuda_studies/ensemble_simulation.hpp"
#include "gtest/gtest.h"

namespace gpu {
__device__ bool unicycle_impl([[maybe_unused]] double t, double const* x,
                              double const* u, double* dx) {
  dx[0] = cos(x[2]) * u[0];
  dx[1] = sin(x[2]) * u[0];
  dx[2] = u[1];
  return true;
}

__device__ fsc::DynamicalSystem unicycle = unicycle_impl;
}  // namespace gpu

namespace cpu {
bool unicycle([[maybe_unused]] double t,
              const Eigen::Ref<const Eigen::VectorXd>& x,
              const Eigen::Ref<const Eigen::VectorXd>& u,
              Eigen::Ref<Eigen::VectorXd> dx) {
  dx[0] = cos(x[2]) * u[0];
  dx[1] = sin(x[2]) * u[0];
  dx[2] = u[1];
  return true;
}
}  // namespace cpu

TEST(TestSimulation, testSimulation) {
  constexpr int kNumSamples = 500;
  constexpr int kNumSteps = 20;
  constexpr int kNx = 3;
  constexpr double kDt = 0.2;
  Eigen::Vector3d x0{10.0, 10.0, EIGEN_PI / 4};

  std::vector<Eigen::MatrixXd> us;
  for (int i = 0; i < kNumSamples; ++i) {
    Eigen::MatrixXd ith_u = Eigen::MatrixXd::Random(kNx, kNumSteps);

    // Move forward only
    ith_u.row(0) += Eigen::RowVectorXd::Ones(kNumSteps);

    // Reduce the magnitude of angular velocity to show a clustered `bundle` of
    // trajectories
    ith_u.row(1) *= 0.5;
    us.emplace_back(ith_u);
  }

  const auto& [ts_result, xs_result, errc_result] =
      fsc::SimulateDynamicalSystemEnsemble(
          gpu::unicycle, 0.0, x0, us, Eigen::VectorXd::Constant(kNumSteps, kDt),
          fsc::Method::kRK4);
  const auto& [ts_expected, xs_expected, errc_expected] =
      fsc::cpu::SimulateDynamicalSystemEnsemble(
          cpu::unicycle, 0.0, x0, us, Eigen::VectorXd::Constant(kNumSteps, kDt),
          fsc::cpu::Method::kRK4);
  ASSERT_EQ(ts_result.size(), ts_expected.size());
  ASSERT_TRUE(ts_result.isApprox(ts_expected)) << ts_result.transpose() << "\n"
                                               << ts_expected.transpose();

  ASSERT_EQ(xs_result.size(), xs_expected.size());
  for (auto it_result = xs_result.cbegin(), it_expected = xs_expected.cbegin();
       it_result != xs_result.cend(); ++it_result, ++it_expected) {
    ASSERT_TRUE(it_result->isApprox(*it_expected));
  }
}
