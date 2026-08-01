from __future__ import annotations
from dataclasses import dataclass, field, fields
from copy import deepcopy
from typing import ClassVar, Any, TYPE_CHECKING

from src.mod.data.traits.error import SimileTraitError
from src.mod.data.traits.trait import Trait
from src.mod.data.traits.traits import (
    OrderableTrait,
    IterableTrait,
    LiteralTrait,
    DomainTrait,
    MinTrait,
    MaxTrait,
    SizeTrait,
    ImmutableTrait,
    TotalOnDomainTrait,
    TotalOnRangeTrait,
    ManyToOneTrait,
    OneToManyTrait,
    EmptyTrait,
    TotalTrait,
    UniqueElementsTrait,
    GenericBoundTrait,
)

if TYPE_CHECKING:
    from src.mod.data.ast_.base import ASTNode
    from src.mod.data.types.base import BaseType


@dataclass
class TraitCollection:
    immutable_trait: ImmutableTrait | None = None

    # Intended for arithmetic use
    orderable_trait: OrderableTrait | None = None
    min_trait: MinTrait | None = None
    max_trait: MaxTrait | None = None

    # Restricting values of a type
    # @deprecated
    # literal_traits: list[LiteralTrait] = field(default_factory=list)

    literal_trait: LiteralTrait | None = None
    domain_trait: DomainTrait | None = None

    # Useful for sets/collections
    iterable_trait: IterableTrait | None = None
    empty_trait: EmptyTrait | None = None
    unique_elements_trait: UniqueElementsTrait | None = None
    size_trait: SizeTrait | None = None

    # On relations, total_trait means every possible pair is enumerated (ie. cartesian product of domain and range)
    # On sets, this just means the set contains every possible element from its domain
    total_trait: TotalTrait | None = None

    # Relation type only - no static validation available for this...
    total_on_domain_trait: TotalOnDomainTrait | None = None
    total_on_range_trait: TotalOnRangeTrait | None = None
    many_to_one_trait: ManyToOneTrait | None = None
    one_to_many_trait: OneToManyTrait | None = None

    # For generic types that may be bound to certain types
    generic_bound_trait: GenericBoundTrait | None = None

    def __post_init__(self):
        self._fill_implicit_traits()

    @property
    def one_to_one(self) -> bool:
        return self.one_to_many_trait is not None and self.many_to_one_trait is not None

    def _fill_implicit_traits(self) -> None:
        """Some traits implicitly encompass others. This fills in that closure.
        Ex. a Literal[1] implies min=1 and max=1.

        Mandatory traits should have been filled in first, restricted traits should be checked after
        """

        # Domain Empty
        if self.domain_trait and len(self.domain_trait.values) == 0:
            self.domain_trait = None

        # Literal Implies a Domain
        if self.literal_trait and self.domain_trait is None:
            self.domain_trait = DomainTrait([self.literal_trait.value])

        # Literal within Domain
        if self.literal_trait and self.domain_trait:
            if self.literal_trait.value not in self.domain_trait.values:
                self.domain_trait = DomainTrait(self.domain_trait.values + [self.literal_trait.value])

        # Orderable Domain without Min
        if self.domain_trait and self.orderable_trait and self.min_trait is None:
            self.min_trait = MinTrait.from_domain_trait(self.domain_trait)

        # Orderable Domain with Min
        if self.domain_trait and self.orderable_trait and self.min_trait:
            min_trait_from_domain = MinTrait.from_domain_trait(self.domain_trait)
            if min_trait_from_domain:
                self.min_trait = self.min_trait.merge(min_trait_from_domain)

        # Orderable Domain without Max
        if self.domain_trait and self.orderable_trait and self.max_trait is None:
            self.max_trait = MaxTrait.from_domain_trait(self.domain_trait)

        # Orderable Domain with Max
        if self.domain_trait and self.orderable_trait and self.max_trait:
            max_trait_from_domain = MaxTrait.from_domain_trait(self.domain_trait)
            if max_trait_from_domain:
                self.max_trait = self.max_trait.merge(max_trait_from_domain)

        # Min implies Order
        if self.min_trait and self.orderable_trait is None:
            self.orderable_trait = OrderableTrait()

        # Max implies Order
        if self.max_trait and self.orderable_trait is None:
            self.orderable_trait = OrderableTrait()

        # Full set
        if self.unique_elements_trait and self.size_trait and self.domain_trait and self.size_trait.size == len(self.domain_trait.values) and self.empty_trait is None:
            self.total_trait = TotalTrait()

        # Empty Size
        if self.size_trait and self.size_trait.size == 0:
            self.empty_trait = EmptyTrait()

        # Non-empty Size
        if self.size_trait and self.size_trait.size > 0:
            self.empty_trait = None

        # Size implies Iterable
        if self.size_trait:
            self.iterable_trait = IterableTrait()

        # Orderable Literal is Min
        if self.literal_trait and self.orderable_trait and self.min_trait is None:
            self.min_trait = MinTrait(value=self.literal_trait.value)

        # Orderable Literal is Max
        if self.literal_trait and self.orderable_trait and self.max_trait is None:
            self.max_trait = MaxTrait(value=self.literal_trait.value)

    def merge(self, other: TraitCollection, prioritize_self_over_other: bool = False) -> TraitCollection:
        """Merge this TraitCollection with another, returning a new TraitCollection.

        Prioritize_self_over_other indicates whether to keep self's traits when both are present."""
        merged = deepcopy(self)

        for trait in fields(merged):
            self_trait = getattr(merged, trait.name)
            other_trait = getattr(other, trait.name)

            if self_trait is None:
                continue
            if other_trait is None or prioritize_self_over_other:
                setattr(merged, trait.name, self_trait)
                continue

            if hasattr(self_trait, "merge"):
                merged_trait = self_trait.merge(other_trait)
                setattr(merged, trait.name, merged_trait)

        merged._fill_implicit_traits()
        return merged

    # TODO make an add_trait like this for merging? (ex. take the lowest of the min)
    def set_trait(self, trait: Trait) -> None:
        """Set a trait on this TraitCollection, modifying it in place.

        Only GenericBoundTrait adds to existing entries, other traits will overwrite any existing trait of the same type.
        """
        match trait:
            case LiteralTrait(_):
                self.literal_trait = trait
            case DomainTrait(_):
                self.domain_trait = trait
            case MinTrait(_):
                self.min_trait = trait
            case MaxTrait(_):
                self.max_trait = trait
            case SizeTrait(_):
                self.size_trait = trait
            case GenericBoundTrait(items):
                if self.generic_bound_trait is not None:
                    self.generic_bound_trait.bound_types.extend(items)
                else:
                    self.generic_bound_trait = trait
            case OrderableTrait():
                self.orderable_trait = trait
            case IterableTrait():
                self.iterable_trait = trait
            case ImmutableTrait():
                self.immutable_trait = trait
            case TotalOnDomainTrait():
                self.total_on_domain_trait = trait
            case TotalOnRangeTrait():
                self.total_on_range_trait = trait
            case ManyToOneTrait():
                self.many_to_one_trait = trait
            case OneToManyTrait():
                self.one_to_many_trait = trait
            case EmptyTrait():
                self.empty_trait = trait
            case TotalTrait():
                self.total_trait = trait
            case UniqueElementsTrait():
                self.unique_elements_trait = trait
            case _:
                raise SimileTraitError(f"Unknown trait: {trait} (failed to set trait on TraitCollection)")
        self._fill_implicit_traits()
        return
