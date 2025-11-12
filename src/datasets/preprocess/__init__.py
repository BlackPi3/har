"""
Dataset preprocessing utilities.

Each dataset-specific preprocessor lives in this package and exposes a small
CLI so it can be invoked directly:

    python -m src.datasets.preprocess.utd_mhad --data-dir datasets/UTD_MHAD
"""
from importlib import import_module
from typing import Any

__all__ = ["UTDMHADPreprocessConfig", "UTDMHADPreprocessor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".utd_mhad", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
