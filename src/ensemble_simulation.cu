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

#include <cstdint>

#include "Eigen/Core"  // IWYU pragma: export
#include "cuda_studies/common.hpp"
#include "cuda_studies/ensemble_simulation.hpp"
#include "thrust/device_vector.h"

namespace fsc {

#define TRY(condition)  \
  do {                  \
    if (!(condition)) { \
      return false;     \
    }                   \
  } while (0)

#define GUARD(condition, res, code) \
  do {                              \
    if ((condition)) {              \
      (res).errc = code;            \
      return res;                   \
    }                               \
  } while (0)
// Declarations
// ============
//
// We put the declarations and the documentation block for the CUDA-specific
// functions here, so that the header file may be pure C++

/**
 * @brief Performs a single step with the Euler method
 *
 * @param system A pointer to the function modelling the dynamical system
 * @param t0 The value of the initial time
 * @param x0 An array containing the initial state
 * @param len_x Number of states in the initial state array
 * @param u An array containing controls in this timestep
 * @param dt The integration timestep
 * @param t_next Pointer to the updated time
 * @param x_next An array to be filled with the updated state
 */
__device__ bool EulerStep(DynamicalSystem system, double t0, double const* x0,
                          int64_t len_x, double const* u, double dt,
                          double* t_next, double* x_next);

/**
 * @brief Performs a single step with the RK4 method
 *
 * @param system A pointer to the function modelling the dynamical system
 * @param t0 The value of the initial time
 * @param x0 An array containing the initial state
 * @param len_x Number of states in the initial state array
 * @param u An array containing controls in this timestep
 * @param dt The integration timestep
 * @param t_next Pointer to the updated time
 * @param x_next An array to be filled with the updated state
 * @return bool
 */
__device__ bool RK4Step(DynamicalSystem system, double t0, double const* x0,
                        int64_t len_x, double const* u, double dt,
                        double* t_next, double* x_next);

/**
 * @brief Solve the dynamical systems equation over a number of timesteps,
 * subject to control inputs
 *
 * @param system A pointer to the function modelling the dynamical system
 * @param t0 The value of the initial time
 * @param x0 An array containing the initial state
 * @param len_x Number of states in the initial state array
 * @param us A flattened array containing a chunk of controls for each
 timestep
 * @param num_steps Number of timesteps
 * @param len_u Number of controls in each chunk
 * @param dt A array containing one integration stepsize for each timestep,
 up
 * to a total of num_steps - 1 stepsizes
 * @param xs A flattened array to be filled with chunks of state, one for
 each
 * timestep
 * @param method Determines if the Euler method or the RK4 method will be
 used
 */

__device__ bool SimulateDynamicalSystem(DynamicalSystem system, double t0,
                                        double const* x0, int64_t len_x,
                                        double const* us, int64_t num_steps,
                                        int64_t len_u, double const* dt,
                                        double* ts, double* xs, Method method);

/**
 * @brief Solve an ensemble of dynamical systems equation over a number of
 * timesteps, subject to control inputs. This overload takes the input samples
 * as a flat array
 *
 * @param system A pointer to the function modelling the dynamical system
 * @param t0 The value of the initial time
 * @param x0 An array containing the initial state
 * @param len_x Number of states in the initial state array
 * @param us A flattened array containing major chunks of controls over all
 * timesteps for each system, each consisting of minor chunks of controls for
 * each timestep
 * @param num_samples The number of systems in the ensemble
 * @param num_steps Number of timesteps
 * @param len_u Number of controls in each chunk
 * @param dt A array containing one integration stepsize for each timestep, up
 * to a total of num_steps - 1 stepsizes
 * @param xs A flattened array to be filled with major chunks of state, one for
 * each system, each consisting of minor chunks of states for each timestep
 * @param method Determines if the Euler method or the RK4 method will be used
 * @return __global__
 */
__global__ void SimulateDynamicalSystemEnsembleKernel(
    DynamicalSystem system, double t, double const* x, int64_t len_x,
    double const* u, int64_t num_samples, int64_t num_steps, int64_t len_u,
    double const* dt, double* ts, double* xs, bool* success, Method method);

// Implementations
// ===============
__device__ bool EulerStep(DynamicalSystem system, double t0, double const* x0,
                          int64_t len_x, double const* u, double dt,
                          double* t_next, double* x_next) {
  Eigen::VectorXd x_step(len_x);
  TRY((*system)(t0, x0, u, x_step.data()));

  const Eigen::Map<const Eigen::VectorXd> x0_v(x0, len_x);
  Eigen::Map<Eigen::VectorXd> x_next_v(x_next, len_x);
  *t_next = t0 + dt;
  x_next_v = x0_v + dt * x_step;
  return true;
}

__device__ bool RK4Step(DynamicalSystem system, double t0, double const* x0,
                        int64_t len_x, double const* u, double dt,
                        double* t_next, double* x_next) {
  constexpr int64_t kOrder = 4;

  Eigen::MatrixXd k(len_x, kOrder);

  // First-order step
  TRY((*system)(t0, x0, u, k.col(0).data()));

  const Eigen::Map<const Eigen::VectorXd> x0_v(x0, len_x);

  // Second-order step
  const double t_midway = t0 + 0.5 * dt;
  Eigen::VectorXd x_op(len_x);
  x_op = x0_v + 0.5 * dt * k.col(0);
  TRY((*system)(t_midway, x_op.data(), u, k.col(1).data()));

  // Third-order step
  x_op = x0_v + 0.5 * dt * k.col(1);
  TRY((*system)(t_midway, x_op.data(), u, k.col(2).data()));

  // Fourth-order step
  *t_next = t0 + dt;
  x_op = x0_v + dt * k.col(2);
  TRY((*system)(*t_next, x_op.data(), u, k.col(3).data()));

  Eigen::Map<Eigen::VectorXd> x_next_v(x_next, len_x);
  x_next_v = x0_v + dt * k * Eigen::Vector4d{1.0, 2.0, 2.0, 1.0} / 6.0;
  return true;
}

__device__ bool SimulateDynamicalSystem(DynamicalSystem system, double t0,
                                        double const* x0, int64_t len_x,
                                        double const* us, int64_t num_steps,
                                        int64_t len_u, double const* dt,
                                        double* ts, double* xs, Method method) {
  // Loop-carried values (t, x)
  Eigen::VectorXd carry = Eigen::Map<const Eigen::VectorXd>(x0, len_x);
  double t_carry = t0;

  // Cache for the computed output at each step is a separate variable in case
  // the dynamical system function's arguments have __restrict__
  for (int64_t k = 0; k < num_steps;
       ++k, us += len_u, ++dt, ++ts, xs += len_x) {
    Eigen::VectorXd next(len_x);
    switch (method) {
      case Method::kEuler:
        TRY(EulerStep(system, t_carry, carry.data(), len_x, us, *dt, &t_carry,
                      next.data()));
        break;
      case Method::kRK4:
        TRY(RK4Step(system, t_carry, carry.data(), len_x, us, *dt, &t_carry,
                    next.data()));

        break;
    }
    carry = next;

    // Copy the loop-carrying variable to the output array
    Eigen::Map<Eigen::VectorXd> xs_v(xs, len_x);
    xs_v = carry;
    *ts = t_carry;
  }
  return true;
}

__global__ void SimulateDynamicalSystemEnsembleKernel(
    DynamicalSystem system, double t, double const* x, int64_t len_x,
    double const* u, int64_t num_samples, int64_t num_steps, int64_t len_u,
    double const* dt, double* ts, double* xs, bool* success, Method method) {
  const int64_t i_system = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_system >= num_samples) {
    return;
  }

  const double* ith_u = u + i_system * num_steps * len_u;
  double* ith_x = xs + i_system * num_steps * len_x;
  success[i_system] = SimulateDynamicalSystem(
      system, t, x, len_x, ith_u, num_steps, len_u, dt, ts, ith_x, method);
}

SimulationResult SimulateDynamicalSystemEnsemble(
    const DynamicalSystem& system, double t0, const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& u, const Eigen::VectorXd& dt,
    Method method) {
  SimulationResult res;
  const int64_t len_x = x0.size();

  GUARD(len_x == 0, res, SimulationErrc::kDimensionsInvalid);

  // Allocate initial conditions in GPU memory
  // =========================================
  thrust::device_vector<double> dev_x0{x0.data(), x0.data() + len_x};

  // Validate input control trajectory
  // =================================
  GUARD(u.empty(), res, SimulationErrc::kNonStarting);
  const auto num_samples = static_cast<int64_t>(u.size());

  // NOTE(Hs293Go): Individual Eigen Matrices are column major, thus control
  // slices are stacked by column
  const auto& u0 = u.front();
  GUARD(u0.size() == 0, res, SimulationErrc::kNonStarting);

  const int64_t len_u = u0.rows();
  const int64_t num_steps = u0.cols();

  // Ensure all samples share common # rows (dimension) and # cols (trajectory
  // length)
  const bool has_inhomogeneous_sample =
      std::any_of(u.cbegin(), u.cend(), [len_u, num_steps](auto&& it) {
        return it.rows() != len_u || it.cols() != num_steps;
      });
  GUARD(has_inhomogeneous_sample, res, SimulationErrc::kDimensionsInconsistent);

  // Allocate input control trajectory in GPU memory
  // ===============================================
  const int64_t u_sample_size = len_u * num_steps;  // Size of each input sample
  thrust::device_vector<double> dev_us(num_samples * u_sample_size);

  // Copy input control trajectory to GPU memory
  // ===========================================
  auto it_u = u.cbegin();  // Source
  const auto sent_u = u.cend();
  auto it_dev_u = dev_us.begin();
  for (; it_u != sent_u; ++it_u, it_dev_u += u_sample_size) {
    thrust::copy_n(it_u->data(), u_sample_size, it_dev_u);
  }

  GUARD(dt.size() != num_steps, res, SimulationErrc::kTimestepsInconsistent);

  GUARD((dt.array() <= 0.0).any(), res, SimulationErrc::kTimestepInvalid);

  thrust::device_vector<double> dev_dt(dt.data(), dt.data() + dt.size());

  // Allocate output state trajectory in GPU memory
  // ==============================================
  thrust::device_vector<double> dev_ts(num_steps);
  const int64_t x_sample_size = len_x * num_steps;
  thrust::device_vector<double> dev_xs(num_samples * x_sample_size);

  // Invoke cuda kernel
  // ==================
  constexpr int64_t kBlockSize = 256;
  const int64_t threads_per_block =
      std::max(1L, (num_samples + kBlockSize - 1) / kBlockSize);

  DynamicalSystem p_system;
  // Copy device function pointer to host side
  cudaMemcpyFromSymbol(&p_system, system, sizeof(DynamicalSystem));

  thrust::device_vector<bool> dev_success(num_samples, false);

  SimulateDynamicalSystemEnsembleKernel<<<threads_per_block, kBlockSize>>>(
      p_system, t0, dev_x0.data().get(), static_cast<int64_t>(dev_x0.size()),
      dev_us.data().get(), num_samples, num_steps, len_u, dev_dt.data().get(),
      dev_ts.data().get(), dev_xs.data().get(), dev_success.data().get(),
      method);
  CUDA_CHECK(cudaDeviceSynchronize());

  const bool has_unsuccessful = std::any_of(
      dev_success.cbegin(), dev_success.cend(), [](auto it) { return !it; });

  GUARD(has_unsuccessful, res, SimulationErrc::kUserAsked);

  // Populate output structure
  // =========================
  res.t = Eigen::VectorXd(num_steps);
  res.x = std::vector<Eigen::MatrixXd>(num_samples,
                                       Eigen::MatrixXd(len_x, num_steps));
  res.errc = SimulationErrc::kSuccess;
  thrust::copy(dev_ts.cbegin(), dev_ts.cend(), res.t.data());

  auto it_x = res.x.begin();
  auto sent_x = res.x.end();
  auto it_dev_x = dev_xs.cbegin();
  for (; it_x != sent_x; ++it_x, it_dev_x += x_sample_size) {
    thrust::copy_n(it_dev_x, x_sample_size, it_x->data());
  }

  return res;
}
}  // namespace fsc
