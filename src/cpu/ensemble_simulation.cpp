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

#include <cstdint>

#include "Eigen/Core"  // IWYU pragma: export
#include "cuda_studies/common.hpp"

namespace fsc::cpu {

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
bool EulerStep(const DynamicalSystem& system, double t0,
               const Eigen::Ref<const Eigen::VectorXd>& x0,
               const Eigen::Ref<const Eigen::VectorXd>& u, double dt,
               double& t_next, Eigen::Ref<Eigen::VectorXd> x_next);

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
bool RK4Step(const DynamicalSystem& system, double t0,
             const Eigen::Ref<const Eigen::VectorXd>& x0,
             const Eigen::Ref<const Eigen::VectorXd>& u, double dt,
             double& t_next, Eigen::Ref<Eigen::VectorXd> x_next);

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

bool SimulateDynamicalSystem(const DynamicalSystem& system, double t0,
                             const Eigen::Ref<const Eigen::VectorXd>& x0,
                             const Eigen::Ref<const Eigen::MatrixXd>& us,
                             const Eigen::Ref<const Eigen::VectorXd>& dt,
                             Eigen::Ref<Eigen::VectorXd> ts,
                             Eigen::Ref<Eigen::MatrixXd> xs, Method method);

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
void SimulateDynamicalSystemEnsembleKernel(
    const DynamicalSystem& system, double t,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const std::vector<Eigen::MatrixXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& dt, Eigen::Ref<Eigen::VectorXd> ts,
    std::vector<Eigen::MatrixXd>& xs, bool& success, Method method);

// Implementations
// ===============
bool EulerStep(const DynamicalSystem& system, double t0,
               const Eigen::Ref<const Eigen::VectorXd>& x0,
               const Eigen::Ref<const Eigen::VectorXd>& u, double dt,
               double& t_next, Eigen::Ref<Eigen::VectorXd> x_next) {
  const auto len_x = x0.size();
  Eigen::VectorXd x_step(len_x);
  TRY(system(t0, x0, u, x_step));

  t_next = t0 + dt;
  x_next = x0 + dt * x_step;
  return true;
}

bool RK4Step(const DynamicalSystem& system, double t0,
             const Eigen::Ref<const Eigen::VectorXd>& x0,
             const Eigen::Ref<const Eigen::VectorXd>& u, double dt,
             double& t_next, Eigen::Ref<Eigen::VectorXd> x_next) {
  constexpr int64_t kOrder = 4;

  const auto len_x = x0.size();
  Eigen::MatrixXd k(len_x, kOrder);

  // First-order step
  TRY(system(t0, x0, u, k.col(0)));

  // Second-order step
  const double t_midway = t0 + 0.5 * dt;
  Eigen::VectorXd x_op(len_x);
  x_op = x0 + 0.5 * dt * k.col(0);
  TRY(system(t_midway, x_op, u, k.col(1)));

  // Third-order step
  x_op = x0 + 0.5 * dt * k.col(1);
  TRY(system(t_midway, x_op, u, k.col(2)));

  // Fourth-order step
  t_next = t0 + dt;
  x_op = x0 + dt * k.col(2);
  TRY(system(t_next, x_op, u, k.col(3)));

  x_next = x0 + dt * k * Eigen::Vector4d(1.0, 2.0, 2.0, 1.0) / 6.0;
  return true;
}

bool SimulateDynamicalSystem(const DynamicalSystem& system, double t0,
                             const Eigen::Ref<const Eigen::VectorXd>& x0,
                             const Eigen::Ref<const Eigen::MatrixXd>& us,
                             const Eigen::Ref<const Eigen::VectorXd>& dt,
                             Eigen::Ref<Eigen::VectorXd> ts,
                             Eigen::Ref<Eigen::MatrixXd> xs, Method method) {
  const auto len_x = x0.size();
  const auto num_steps = us.cols();
  // Loop-carried values (t, x)
  Eigen::VectorXd carry = x0;
  double t_carry = t0;

  // Cache for the computed output at each step is a separate variable in case
  // the dynamical system function's arguments have __restrict__
  for (int64_t k = 0; k < num_steps; ++k) {
    Eigen::VectorXd next(len_x);
    switch (method) {
      case Method::kEuler:
        TRY(EulerStep(system, t_carry, carry, us.col(k), dt[k], t_carry, next));
        break;
      case Method::kRK4:
        TRY(RK4Step(system, t_carry, carry, us.col(k), dt[k], t_carry, next));

        break;
    }
    carry = next;

    // Copy the loop-carrying variable to the output array
    xs.col(k) = carry;
    ts[k] = t_carry;
  }
  return true;
}

SimulationResult SimulateDynamicalSystemEnsemble(
    const DynamicalSystem& system, double t0, const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& u, const Eigen::VectorXd& dt,
    Method method) {
  SimulationResult res;
  const int64_t len_x = x0.size();

  GUARD(len_x == 0, res, SimulationErrc::kDimensionsInvalid);

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

  GUARD(dt.size() != num_steps, res, SimulationErrc::kTimestepsInconsistent);

  GUARD((dt.array() <= 0.0).any(), res, SimulationErrc::kTimestepInvalid);

  // Copy device function pointer to host side
  res.t.resize(num_steps);
  res.x = std::vector<Eigen::MatrixXd>(num_samples,
                                       Eigen::MatrixXd(len_x, num_steps));
  for (int64_t i = 0; i < num_samples; ++i) {
    const auto& us = u[i];
    GUARD(!SimulateDynamicalSystem(system, t0, x0, us, dt, res.t, res.x[i],
                                   method),
          res, SimulationErrc::kUserAsked);
  }

  return res;
}
}  // namespace fsc::cpu
