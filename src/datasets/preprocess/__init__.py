"""
Dataset preprocessing utilities.

Each dataset-specific preprocessor lives in this package and exposes a small
CLI so it can be invoked directly:

    python -m src.datasets.preprocess.utd --data-dir datasets/utd
"""
from importlib import import_module
from typing import Any

_EXPORTS = {
    "UTDPreprocessConfig": ".utd",
    "UTDPreprocessor": ".utd",
    "MMFitPreprocessConfig": ".mmfit",
    "MMFitPreprocessor": ".mmfit",
    "NTUPreprocessConfig": ".ntu",
    "NTUPreprocessor": ".ntu",
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path:
        module = import_module(module_path, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
