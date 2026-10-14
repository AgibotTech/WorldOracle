"""Public ACWMBench API with lazy ML dependency loading."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import ACWMBenchmark

    EmbodiedWorldModelBenchmark = ACWMBenchmark

# Names are provided dynamically by __getattr__ to avoid importing torch for
# lightweight metric-only workflows.
# pylint: disable=undefined-all-variable
__all__ = ["ACWMBenchmark", "EmbodiedWorldModelBenchmark"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runner import ACWMBenchmark

        return ACWMBenchmark
    raise AttributeError(name)
