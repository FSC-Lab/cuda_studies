project(${CMAKE_PROJECT_NAME}_benchmarks)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_WERROR OFF)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.0
  SYSTEM FIND_PACKAGE_ARGS NAMES benchmark CONFIG 1.8.0)

FetchContent_MakeAvailable(benchmark)

add_executable(${PROJECT_NAME}_bm_simulation src/benchmark_simulation.cu)
target_link_libraries(
  ${PROJECT_NAME}_bm_simulation
  PRIVATE benchmark::benchmark benchmark::benchmark_main ensemble_simulation
          cpu_ensemble_simulation)
set_target_properties(${PROJECT_NAME}_bm_simulation
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
