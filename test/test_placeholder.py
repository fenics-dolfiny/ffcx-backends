import importlib


def test() -> None:
    importlib.import_module("ffcx_backends")
    assert True
