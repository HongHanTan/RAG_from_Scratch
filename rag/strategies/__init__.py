"""Strategy registry.

One place that maps a CLI name to a strategy, so adding a strategy does not
mean touching the pipeline or the CLI.
"""

from __future__ import annotations

from rag.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyResult,
    degrade_to_direct,
)
from rag.strategies.direct import DirectStrategy
from rag.strategies.multi_query import MultiQueryStrategy
from rag.strategies.rag_fusion import RagFusionStrategy
from rag.strategies.step_back import StepBackStrategy

__all__ = [
    "Strategy",
    "StrategyContext",
    "StrategyResult",
    "degrade_to_direct",
    "STRATEGY_NAMES",
    "get_strategy",
]

_REGISTRY: dict[str, type] = {
    DirectStrategy.name: DirectStrategy,
    MultiQueryStrategy.name: MultiQueryStrategy,
    RagFusionStrategy.name: RagFusionStrategy,
    StepBackStrategy.name: StepBackStrategy,
}

STRATEGY_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def get_strategy(name: str, **options) -> Strategy:
    """Build a strategy by name. Unknown names raise rather than defaulting."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy: {name} (choose one of {', '.join(sorted(_REGISTRY))})"
        ) from None
    return cls(**options)
