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

#ifndef CUDA_STUDIES_CPU_ENSEMBLE_SIMULATION_HPP_
#define CUDA_STUDIES_CPU_ENSEMBLE_SIMULATION_HPP_

#include <functional>
#include <memory>
#include <vector>

#include "Eigen/Dense"

namespace fsc::cpu {
using DynamicalSystem = std::function<bool(
    double, const Eigen::Ref<const Eigen::VectorXd>&,
    const Eigen::Ref<const Eigen::VectorXd>&, Eigen::Ref<Eigen::VectorXd>)>;

enum class Method { kEuler, kRK4 };

struct SimulationResult {
  Eigen::VectorXd t;
  std::vector<Eigen::MatrixXd> x;
  std::error_code errc;
};

/**
 * @brief Solve an ensemble of dynamical systems equation over a number of
 * timesteps, subject to control inputs.
 *
 * @param system A pointer to the function modelling the dynamical system
 * @param t0 The value of the initial time
 * @param x0 An array containing the initial state
 * @param u A vector of arrays containing controls over all timesteps for each
 * system, each consisting of columnwise slices of controls for each timestep
 * @param dt An array containing the timesteps for each timestep. This is a
 * single array, common to every system
 * @param method A enumerator toggling usage of Euler or RK4 methods
 * @return A SimulationResult structure containing the output time and state
 * trajectory and an error code
 */
SimulationResult SimulateDynamicalSystemEnsemble(
    const DynamicalSystem& system, double t0, const Eigen::VectorXd& x0,
    const std::vector<Eigen::MatrixXd>& u, const Eigen::VectorXd& dt,
    Method method);

class DynamicalSystemEnsemble {
 public:
  DynamicalSystemEnsemble(const DynamicalSystem& system, Method method);

  ~DynamicalSystemEnsemble();

  SimulationResult simulate(double t0, const Eigen::VectorXd& x0,
                            const std::vector<Eigen::MatrixXd>& u,
                            const Eigen::VectorXd& dt);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};
}  // namespace fsc::cpu

#endif  // CUDA_STUDIES_CPU_ENSEMBLE_SIMULATION_HPP_
