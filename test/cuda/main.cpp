#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <sstream>

int
main()
{
  // CUDA kernel code as a string
  std::ifstream t("test/cuda/sample_integral.cu");
  if (t.peek() == std::ifstream::traits_type::eof())
    throw std::runtime_error("File not found.");

  std::stringstream buffer;
  buffer << t.rdbuf();
  auto kernel_code = buffer.str();

  std::cout << kernel_code << std::endl;

  nvrtcProgram prog;
  nvrtcResult res;

  // Create NVRTC program
  res = nvrtcCreateProgram(
    &prog, kernel_code.c_str(), "add_kernel.cu", 0, nullptr, nullptr);
  if (res != NVRTC_SUCCESS) {
    std::cerr << "Failed to create program: " << nvrtcGetErrorString(res)
              << "\n";
    return 1;
  }

  // Compile program
  res = nvrtcCompileProgram(prog, 0, nullptr);
  if (res != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    std::cerr << "Compilation failed:\n" << log << "\n";
    return 1;
  }

  // Get PTX
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, &ptx[0]);

  // std::cout << "PTX code:\n" << ptx << "\n";

  // Clean up
  nvrtcDestroyProgram(&prog);
  return 0;
}
