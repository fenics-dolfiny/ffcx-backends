cmake_minimum_required(VERSION 3.10)
project(nvrtc_compiler)

# find nvrtc
execute_process(
    COMMAND python ${CMAKE_SOURCE_DIR}/find-nvrtc.py
    OUTPUT_VARIABLE NVRTC_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "Found nvrtc at ${NVRTC_ROOT}")

add_executable(nvrtc_compiler main.cpp)
set_property(TARGET nvrtc_compiler PROPERTY CXX_STANDARD 20)

target_link_libraries(nvrtc_compiler PUBLIC ${NVRTC_ROOT}/lib/libnvrtc.so.12)
target_include_directories(nvrtc_compiler PUBLIC ${NVRTC_ROOT}/include)
