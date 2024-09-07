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

#ifndef CUDA_STUDIES_COMMON_HPP_
#define CUDA_STUDIES_COMMON_HPP_

#include <cstdint>
#include <string>
#include <system_error>

#include "cuda_runtime_api.h"  // IWYU pragma: export

void CudaCheckImpl(char const* condition, cudaError_t code, char const* file,
                   int64_t line);

#define CUDA_CHECK(...)                                             \
  do {                                                              \
    if (__VA_ARGS__ != cudaSuccess) {                               \
      CudaCheckImpl(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__); \
    }                                                               \
  } while (0)

namespace fsc {
enum class SimulationErrc {
  kSuccess = 0,
  kNonStarting = 1,
  kUserAsked = 2,
  kTimestepInvalid = 3,
  kDimensionsInvalid = 4,
  kTimestepsInconsistent = 5,
  kDimensionsInconsistent = 6,
};
}  // namespace fsc

namespace std {
template <>
struct is_error_code_enum<::fsc::SimulationErrc> : true_type {};
}  // namespace std

namespace fsc {

namespace detail {
class SimulationErrcCategory final : public std::error_category {
 public:
  [[nodiscard]] const char* name() const noexcept final {
    return "ControllerError";
  }

  [[nodiscard]] std::string message(int c) const final;
};
}  // namespace detail

extern inline const detail::SimulationErrcCategory&
GetSimulationErrcCategory() {
  static detail::SimulationErrcCategory c;
  return c;
}

inline std::error_code make_error_code(fsc::SimulationErrc errc) {
  return {static_cast<int>(errc), GetSimulationErrcCategory()};
}
}  // namespace fsc
#endif  // CUDA_STUDIES_COMMON_HPP_
