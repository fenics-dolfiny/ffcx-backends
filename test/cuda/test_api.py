import re
from pathlib import Path

import ffcx.main


def test_cuda_backend() -> None:
    """Test CUDA backend."""

    opts = "--language ffcx_backends.cuda --scalar_type float64"
    directory = Path(__file__).parent.parent
    assert ffcx.main.main([str(directory / "poisson.py"), *opts.split(" ")]) == 0


def test_cuda_interpolation(tmp_path: Path) -> None:
    """Interpolation kernels are device functions, defined before the kernels calling them."""
    opts = ["--language", "ffcx_backends.cuda", "--scalar_type", "float64", "-d", str(tmp_path)]
    directory = Path(__file__).parent.parent
    assert ffcx.main.main([str(directory / "interpolation.py"), *opts]) == 0

    source = (tmp_path / "interpolation.cu").read_text()
    kernel = r"(tabulate_tensor_expression_\w+)\("
    definitions = {
        m.group(1): m.start() for m in re.finditer(rf"static __device__\nvoid {kernel}", source)
    }
    calls = [(m.group(1), m.start()) for m in re.finditer(rf"(?<!void ){kernel}", source)]
    # One kernel each for u**2, shared by a form and the nested expression,
    # u * v and the operand of the nested expression.
    assert len(definitions) == 3
    # From both integrals and the operand of the nested expression. The nested
    # expression itself is not generated, CUDA expressions are not supported.
    assert len(calls) == 3
    for name, position in calls:
        assert definitions[name] < position
