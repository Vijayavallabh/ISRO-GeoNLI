"""
Global registries using a name -> callable (factory) pattern.
This enables plugin-style extension without modifying core code.
"""

from typing import Callable, Dict, Any

# ---------------------------------------------------------------------------
# Internal registries
# ---------------------------------------------------------------------------

_vlm_registry: Dict[str, Callable[..., Any]] = {}
_segmenter_registry: Dict[str, Callable[..., Any]] = {}
_task_registry: Dict[str, Callable[..., Any]] = {}
_dataset_registry: Dict[str, Callable[..., Any]] = {}
_prompt_registry: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------

def register_vlm(name: str):
    """Register a VLM factory under ``name``."""
    def decorator(cls_or_fn):
        _vlm_registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def register_segmenter(name: str):
    """Register a segmenter factory under ``name``."""
    def decorator(cls_or_fn):
        _segmenter_registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def register_task(name: str):
    """Register a task factory under ``name``."""
    def decorator(cls_or_fn):
        _task_registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def register_dataset(name: str):
    """Register a dataset factory under ``name``."""
    def decorator(cls_or_fn):
        _dataset_registry[name] = cls_or_fn
        return cls_or_fn
    return decorator


def register_prompt(name: str, template: str):
    """Register a prompt template string under ``name``."""
    _prompt_registry[name] = template
    return template


# ---------------------------------------------------------------------------
# Lookup / factory functions
# ---------------------------------------------------------------------------

def get_vlm(name: str, **kwargs):
    if name not in _vlm_registry:
        raise KeyError(
            f"VLM '{name}' not found. "
            f"Registered: {list(_vlm_registry.keys())}"
        )
    return _vlm_registry[name](**kwargs)


def get_segmenter(name: str, **kwargs):
    if name not in _segmenter_registry:
        raise KeyError(
            f"Segmenter '{name}' not found. "
            f"Registered: {list(_segmenter_registry.keys())}"
        )
    return _segmenter_registry[name](**kwargs)


def get_task(name: str, **kwargs):
    if name not in _task_registry:
        raise KeyError(
            f"Task '{name}' not found. "
            f"Registered: {list(_task_registry.keys())}"
        )
    return _task_registry[name](**kwargs)


def get_dataset(name: str, **kwargs):
    if name not in _dataset_registry:
        raise KeyError(
            f"Dataset '{name}' not found. "
            f"Registered: {list(_dataset_registry.keys())}"
        )
    return _dataset_registry[name](**kwargs)


def get_prompt(name: str) -> str:
    if name not in _prompt_registry:
        raise KeyError(
            f"Prompt '{name}' not found. "
            f"Registered: {list(_prompt_registry.keys())}"
        )
    return _prompt_registry[name]


def list_registry() -> Dict[str, Any]:
    return {
        "vlm": list(_vlm_registry.keys()),
        "segmenter": list(_segmenter_registry.keys()),
        "task": list(_task_registry.keys()),
        "dataset": list(_dataset_registry.keys()),
        "prompt": list(_prompt_registry.keys()),
    }
