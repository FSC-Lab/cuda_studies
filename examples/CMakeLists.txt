project(${CMAKE_PROJECT_NAME}_examples)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.0
  SYSTEM FIND_PACKAGE_ARGS NAMES pybind11 CONFIG 2.13.0)

FetchContent_MakeAvailable(pybind11)

add_executable(run_simulation src/run_simulation.cu)
set_target_properties(run_simulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(run_simulation PRIVATE common ensemble_simulation
                                             pybind11::embed)
