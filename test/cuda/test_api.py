import ffcx.main
from pathlib import Path


def test_cuda_backend():
    """Test CUDA backend."""

    opts = "--language ffcx_backends.cuda --scalar_type float64"
    dir = Path(__file__).parent.parent
    assert ffcx.main.main([str(dir / "poisson.py"), *opts.split(" ")]) == 0
