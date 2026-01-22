import importlib.util
import os
import pathlib
import subprocess


def test_demo_nvrtc():
    """Test CUDA program compile with NVRTC."""

    spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    nvrtc_dir = pathlib.Path(spec.submodule_search_locations[0])

    cxx = os.environ.get("CXX", "c++")
    subprocess.check_call(
        [
            cxx,
            f"-I{nvrtc_dir}/include",
            f"-L{nvrtc_dir}/lib/",
            "-o",
            "nvrtc_test",
            "test/cuda/main.cpp",
            "-l:libnvrtc.so.12",
        ]
    )
    subprocess.check_call(
        ["./nvrtc_test"], env={"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{nvrtc_dir}/lib"}
    )
