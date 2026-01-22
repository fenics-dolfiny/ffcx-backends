from pathlib import Path

import ffcx.main


def test_cuda_backend():
    """Test CUDA backend."""

    opts = "--language ffcx_backends.cuda --scalar_type float64"
    directory = Path(__file__).parent.parent
    assert ffcx.main.main([str(directory / "poisson.py"), *opts.split(" ")]) == 0
