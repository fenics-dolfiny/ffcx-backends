cmake_minimum_required(VERSION 3.14)
project(ffcx-backends-test-cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FeatureSummary)
option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(
    ENABLE_CLANG_TIDY
    ENABLE_CLANG_TIDY
    "Run clang-tidy while building"
)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-error=unused-parameter
)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG
        52eb8108c5bdec04579160ae17225d66034bd723 # 1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

execute_process(
    COMMAND python ${CMAKE_SOURCE_DIR}/find-ffcx.py
    OUTPUT_VARIABLE FFCX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

add_custom_target(
    poisson_kernel
    COMMAND
        ffcx --language ffcx_backends.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../poisson.py -d ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../poisson.py
    COMMENT "Compile poisson.py using FFCx"
)

add_executable(kernel kernel.cpp)
add_dependencies(kernel poisson_kernel)

if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
    set_target_properties(
        kernel
        PROPERTIES
            CXX_CLANG_TIDY
                "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy"
    )
endif()

target_include_directories(kernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(kernel PUBLIC ${FFCX_PATH}/codegeneration)
target_link_libraries(kernel GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(kernel)
