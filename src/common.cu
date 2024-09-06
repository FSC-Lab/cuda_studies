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

#include <cstdio>

#include "cuda_studies/common.hpp"

void CudaCheckImpl(char const* condition, cudaError_t code, char const* file,
                   int64_t line) {
  fprintf(stderr, "%s:%ld: %s failed with message: %s \n", file, line,
          condition, cudaGetErrorString(code));
  exit(code);
}

std::string fsc::detail::SimulationErrcCategory::message(int c) const {
  switch (static_cast<fsc::SimulationErrc>(c)) {
    case fsc::SimulationErrc::kSuccess:
      return "simulation successful";
    case fsc::SimulationErrc::kNonStarting:
      return "Simulation not starting due to empty inputs";
    case fsc::SimulationErrc::kUserAsked:
      return "user requested exit";
    case fsc::SimulationErrc::kTimestepInvalid:
      return "timestep is invalid";
    case fsc::SimulationErrc::kDimensionsInvalid:
      return "input/state dimensions are invalid";
    case fsc::SimulationErrc::kTimestepsInconsistent:
      return "number of timesteps are inconsistent";
    case fsc::SimulationErrc::kDimensionsInconsistent:
      return "input dimensions are inconsistent";

    default:
      return "Invalid error code";
  }
}
