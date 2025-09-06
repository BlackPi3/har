# MMFit Dataset Modularization - Migration Guide

## What Changed

The `mmfit_data.py` file has been refactored into a modular structure for better maintainability and reusability.

## New Structure

```
datasets/
├── common/                    # Shared components across datasets
│   ├── __init__.py           # Common utilities export
│   ├── samplers.py           # RandomStridedSampler, SequentialStridedSampler
│   ├── utils.py              # File loading utilities
│   └── base_dataset.py       # BaseHARDataset class with common functionality
├── mmfit/                    # MMFit-specific modules
│   ├── __init__.py           # MMFit exports
│   ├── constants.py          # ACTIONS, subject IDs, file patterns
│   ├── loaders.py            # load_modality, load_labels
│   ├── dataset.py            # MMFit Dataset class
│   └── factory.py            # build_mmfit_datasets
└── mmfit_data.py             # Original file (kept for now, but deprecated)
```

## Migration for Existing Code

### Old imports:
```python
from datasets.mmfit_data import MMFit, build_mmfit_datasets
from datasets.mmfit_data import RandomStridedSampler, SequentialStridedSampler
```

### New imports:
```python
from datasets import MMFit, build_mmfit_datasets  # Still works via datasets/__init__.py
from datasets.common import RandomStridedSampler, SequentialStridedSampler
```

### Or more explicit:
```python
from datasets.mmfit import MMFit, build_mmfit_datasets
from datasets.common import RandomStridedSampler, SequentialStridedSampler
```

## What Moved Where

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `MMFit` class | `datasets.mmfit.dataset` | Enhanced with better error handling |
| `RandomStridedSampler`, `SequentialStridedSampler` | `datasets.common.samplers` | Can be reused by other datasets |
| `load_modality`, `load_labels` | `datasets.mmfit.loaders` | MMFit-specific loaders |
| `build_mmfit_datasets` | `datasets.mmfit.factory` | Factory function for dataset creation |
| `unfold` function | `src.inference` | Renamed to `unfold_predictions` |
| ACTIONS, subject lists | `datasets.mmfit.constants` | All constants in one place |

## Benefits

1. **Single Responsibility**: Each module has one clear purpose
2. **Reusability**: Common components can be used by other datasets
3. **Better Testing**: Individual components can be tested in isolation
4. **Maintainability**: Changes to one aspect don't affect others
5. **Clear Dependencies**: Import structure shows what depends on what

## Next Steps

1. **Test thoroughly**: Make sure all existing functionality works
2. **Update other code**: Gradually migrate imports in other files
3. **Remove legacy**: After migration, remove `mmfit_data.py`
4. **Extend pattern**: Apply similar structure to other datasets (MHAD, NTU, etc.)

## Backward Compatibility

The old imports still work through `datasets/__init__.py`, so existing code continues to function without immediate changes.
