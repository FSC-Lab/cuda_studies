project(${CMAKE_PROJECT_NAME}_tests)

include(FetchContent)

FetchContent_Declare(
  GTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.12.0
  SYSTEM FIND_PACKAGE_ARGS NAMES GTest CONFIG 1.12.0)

FetchContent_MakeAvailable(GTest)

include(GoogleTest)
add_executable(${PROJECT_NAME} src/test_basic_kernel_launch.cu
                               src/test_function_pointer.cu)
target_compile_options(
  ${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012,20011>
                         -Wno-cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE GTest::gtest GTest::gtest_main)
gtest_discover_tests(${PROJECT_NAME})

add_executable(${PROJECT_NAME}_our_api src/test_simulation.cu)
target_link_libraries(
  ${PROJECT_NAME}_our_api PRIVATE ensemble_simulation cpu_ensemble_simulation
                                  GTest::gtest GTest::gtest_main)
set_target_properties(${PROJECT_NAME}_our_api
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
gtest_discover_tests(${PROJECT_NAME}_our_api)
