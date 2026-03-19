import subprocess
from pathlib import Path

import pytest

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


def test_compiler_bad_source(nvrtc_compiler):
    with pytest.raises(Exception) as error:
        subprocess.check_call(
            ["./nvrtc_compiler", cuda_dir / "not_a_file.cu"],
            cwd=build_dir,
        )
        assert "Could not read file" in error.value


def test_compiler_help(nvrtc_compiler) -> None:
    subprocess.check_call(
        ["./nvrtc_compiler", "--help"],
        cwd=build_dir,
    )


def test_compiler_arg_count(nvrtc_compiler) -> None:
    with pytest.raises(Exception) as error:
        subprocess.check_call(
            ["./nvrtc_compiler", "a", "b"],
            cwd=build_dir,
        )
        assert "Usage:" in error.value


def test_demo_nvrtc(nvrtc_compiler):
    cuda = Path(__file__).parent
    build = cuda / "nvrtc_compiler" / "build"
    source = cuda / "sample_integral.cu"

    subprocess.check_call(
        ["./nvrtc_compiler", source],
        cwd=build,
    )
