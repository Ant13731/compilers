from dataclasses import dataclass

from src.mod.data.traits.base import BaseTrait, SimileLiteralAsPython


@dataclass(frozen=True)
class TreatAsExprTrait(BaseTrait):
    pass
