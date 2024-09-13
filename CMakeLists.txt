# Demand even higher cmake version for experimental code
cmake_minimum_required(VERSION 3.26.0)
project(fsc_cuda_studies LANGUAGES CXX CUDA)

option(BUILD_EXAMPLES "Toggles building examples" ON)
option(BUILD_BENCHMARKS "Toggles building benchmarks" ON)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  SYSTEM FIND_PACKAGE_ARGS NAMES Eigen3 CONFIG 3.4.0)

FetchContent_MakeAvailable(Eigen3)

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
add_library(common SHARED src/common.cu)
target_compile_features(common PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress
                                     20012> -Wno-cpp)
target_include_directories(common PUBLIC include)
set_target_properties(common PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_library(ensemble_simulation src/ensemble_simulation.cu)
target_include_directories(ensemble_simulation PUBLIC include)

set_target_properties(ensemble_simulation PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                     ON)
target_link_libraries(ensemble_simulation PUBLIC Eigen3::Eigen common)

add_library(cpu_ensemble_simulation src/cpu/ensemble_simulation.cpp)
target_compile_features(cpu_ensemble_simulation PUBLIC cxx_std_17)
target_include_directories(cpu_ensemble_simulation PUBLIC include)
target_link_libraries(cpu_ensemble_simulation PUBLIC Eigen3::Eigen)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
