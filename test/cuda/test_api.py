from pathlib import Path

import ffcx.main


def test_cuda_backend():
    """Test CUDA backend."""

    opts = "--language ffcx_backends.cuda --scalar_type float64"
    dirname = Path(__file__).parent.parent
    assert ffcx.main.main([str(dirname / "poisson.py"), *opts.split(" ")]) == 0
