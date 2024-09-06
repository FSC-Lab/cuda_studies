
#include "Eigen/Dense"
#include "cuda_studies/common.hpp"
#include "cuda_studies/ensemble_simulation.hpp"
#include "pybind11/embed.h"
#include "pybind11/numpy.h"

__device__ bool unicycle_impl([[maybe_unused]] double t, double const* x,
                              double const* u, double* dx) {
  dx[0] = cos(x[2]) * u[0];
  dx[1] = sin(x[2]) * u[0];
  dx[2] = u[1];
  return true;
}

__device__ fsc::DynamicalSystem unicycle = unicycle_impl;

int main() {
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

  const auto& [ts, xs, errc] = fsc::SimulateDynamicalSystemEnsemble(
      unicycle, 0.0, x0, us, Eigen::VectorXd::Constant(kNumSteps, kDt),
      fsc::Method::kRK4);

  namespace py = pybind11;
  using namespace pybind11::literals;  // NOLINT
  py::scoped_interpreter interp;

  auto plt = py::module::import("matplotlib.pyplot");

  constexpr auto kAlphaRange = 0.2;
  // Only do plotting if simulation succeeds
  if (errc == fsc::SimulationErrc::kSuccess) {
    auto plot = plt.attr("plot");
    plot(x0[0], x0[1], "ro", "label"_a = "Initial Point");
    for (auto i = 0U; i < xs.size(); ++i) {
      const Eigen::VectorXd x = xs[i].row(0);
      const Eigen::VectorXd y = xs[i].row(1);
      const py::array xv{x.size(), x.data()};
      const py::array yv{y.size(), y.data()};
      plot(xv, yv, "color"_a = "C1",
           "alpha"_a = kAlphaRange * i / static_cast<double>(xs.size()));
      plot(x[x.size() - 1], y[y.size() - 1], "bo", "alpha"_a = 0.5);
    }
    plt.attr("legend")();
  }
  plt.attr("xlabel")("X (m)");
  plt.attr("ylabel")("Y (m)");
  plt.attr("title")(
      "Visualizing {} sampled trajectories\nSimulator message: {}"_s.format(
          kNumSamples, errc.message()));
  plt.attr("show")();
}
