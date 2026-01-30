cmake_minimum_required(VERSION 3.14)

project(my_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(my_test test_example.cpp)
target_link_libraries(my_test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(my_test)