#include <algorithm>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <sstream>
#include <vector>

void
print_help(std::ostream& os)
{
  os << "\n";
  os << "Usage: \n";
  os << "  compile [file] [flags] \n";
  os << " Flags: \n";
  os << "     --help Print this information";
  os << "\n";
  os << std::endl;
}

int
quit(int exit_code)
{
  print_help(exit_code == EXIT_SUCCESS ? std::cout : std::cerr);
  return exit_code;
}

std::string
nvrtc_compile(const std::string& source)
{
  nvrtcProgram prog;
  nvrtcResult res;

  res = nvrtcCreateProgram(
    &prog, source.c_str(), "_source.cu", 0, nullptr, nullptr);
  if (res != NVRTC_SUCCESS) {
    std::cerr << "Failed to create program: " << nvrtcGetErrorString(res)
              << "\n";
    quit(EXIT_FAILURE);
  }

  res = nvrtcCompileProgram(prog, 0, nullptr);
  if (res != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    std::cerr << "Compilation failed:\n" << log << "\n";
    quit(EXIT_FAILURE);
  }

  // Get PTX
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, &ptx[0]);

  // Clean up
  nvrtcDestroyProgram(&prog);

  return ptx;
}

int
main(int argc, char* argv[])
{
  if (argc < 2)
    return quit(EXIT_FAILURE);

  std::vector<std::string> args;
  for (int i = 1; i < argc; i++)
    args.emplace_back(argv[i]);

  if (std::ranges::find(args, "--help") != args.end())
    return quit(EXIT_SUCCESS);

  if (args.size() != 1)
    return quit(EXIT_FAILURE);

  std::string path_input = args[0];

  std::ifstream t(path_input);
  if (t.peek() == std::ifstream::traits_type::eof()) {
    std::cerr << "Could not read file: " << path_input << std::endl;
    return EXIT_FAILURE;
  }

  std::stringstream buffer;
  buffer << t.rdbuf();
  auto kernel_code = buffer.str();

  // TODO: add flag to control output
  // std::cout << kernel_code << std::endl;

  auto ptx = nvrtc_compile(kernel_code);

  // TODO: add flag to control output
  // std::cout << "PTX code:\n" << ptx << "\n";

  return EXIT_SUCCESS;
}
