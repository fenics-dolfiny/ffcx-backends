import subprocess
from pathlib import Path

import ffcx.main
import pytest

pytest.importorskip("nvidia.cuda_nvrtc", reason="NVRTC not available on all platforms.")

cuda_dir = cuda = Path(__file__).parent
build_dir = cuda / "nvrtc_compiler" / "build"


@pytest.fixture
def nvrtc_compiler() -> None:
    build_dir.mkdir(exist_ok=True)
    subprocess.check_call(
        [
            "cmake",
            "..",
        ],
        cwd=build_dir,
    )
    subprocess.check_call(["make"], cwd=build_dir)


def test_compiler_bad_source(nvrtc_compiler: None) -> None:
    with pytest.raises(Exception) as error:
        subprocess.check_call(
            ["./nvrtc_compiler", cuda_dir / "not_a_file.cu"],
            cwd=build_dir,
        )
        assert "Could not read file" in str(error.value)


def test_compiler_help(nvrtc_compiler: None) -> None:
    subprocess.check_call(
        ["./nvrtc_compiler", "--help"],
        cwd=build_dir,
    )


def test_compiler_arg_count(nvrtc_compiler: None) -> None:
    with pytest.raises(Exception) as error:
        subprocess.check_call(
            ["./nvrtc_compiler", "a", "b"],
            cwd=build_dir,
        )
        assert "Usage:" in str(error.value)


def test_sample_integral(nvrtc_compiler: None) -> None:
    cuda = Path(__file__).parent
    build = cuda / "nvrtc_compiler" / "build"
    source = cuda / "sample_integral.cu"

    subprocess.check_call(
        ["./nvrtc_compiler", source],
        cwd=build,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_poisson(nvrtc_compiler: None, dtype: str) -> None:
    opts = f"--language ffcx_backends.cuda --scalar_type {dtype}"
    directory = Path(__file__).parent.parent
    assert ffcx.main.main([str(directory / "poisson.py"), *opts.split(" ")]) == 0

    cuda = Path(__file__).parent
    build = cuda / "nvrtc_compiler" / "build"
    source = cuda / "sample_integral.cu"

    subprocess.check_call([build / "nvrtc_compiler", source], cwd=build)
