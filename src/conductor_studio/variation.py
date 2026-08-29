"""The four-variant assignment policy.

This module is intentionally independent of Conductor Core.  It is the place
where Studio decides which values are durable session inputs; the adapter is
responsible for translating those values to Core's request type.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import Capability, SessionManifest, SessionSettings, VariantSlot

SLOT_IDS = VariantSlot.SLOT_IDS
TEMPERATURES = VariantSlot.TEMPERATURES


@dataclass(frozen=True)
class VariantAssignment:
    """One immutable assignment made before a provider call is started."""

    slot_id: str
    temperature: float
    seed: int | None
    temperature_capability: Capability
    seed_capability: Capability


def _four_seeds(seed_factory: Callable[[], int] | None) -> list[int]:
    factory = seed_factory or (lambda: secrets.randbelow(2**31))
    return [factory() for _ in SLOT_IDS]


def assignments_for(
    *,
    seeds: Iterable[int | None] | None = None,
    temperature_capability: Capability = Capability.SUPPORTED,
    seed_capability: Capability = Capability.UNSUPPORTED,
    seed_factory: Callable[[], int] | None = None,
) -> tuple[VariantAssignment, ...]:
    """Return exactly four fixed assignments in display order.

    The current pinned Core contract has no seed field.  Callers should leave
    ``seed_capability`` unsupported (the default), which makes that omission
    explicit instead of claiming reproducibility Studio cannot provide.
    """

    seed_values = (
        list(seeds)
        if seeds is not None
        else (
            _four_seeds(seed_factory)
            if seed_capability is Capability.SUPPORTED
            else [None] * len(SLOT_IDS)
        )
    )
    if len(seed_values) != len(SLOT_IDS):
        raise ValueError("exactly four seed assignments are required")
    assignments: list[VariantAssignment] = []
    for index, slot_id in enumerate(SLOT_IDS):
        effective_seed = (
            seed_values[index] if seed_capability is Capability.SUPPORTED else None
        )
        assignments.append(
            VariantAssignment(
                slot_id=slot_id,
                temperature=TEMPERATURES[index],
                seed=effective_seed,
                temperature_capability=temperature_capability,
                seed_capability=seed_capability,
            )
        )
    return tuple(assignments)


def create_manifest(
    settings: SessionSettings,
    *,
    session_id: str | None = None,
    title: str | None = None,
    seeds: Iterable[int | None] | None = None,
    temperature_capability: Capability = Capability.SUPPORTED,
    # Unsupported is truthful for the currently pinned Core revision.
    seed_capability: Capability = Capability.UNSUPPORTED,
    studio_version: str = "0.1.0",
    now=None,
) -> SessionManifest:
    """Create a manifest with assignments persisted before generation starts."""

    seed_values = (
        list(seeds)
        if seeds is not None
        else (
            _four_seeds(None)
            if seed_capability is Capability.SUPPORTED
            else [None] * len(SLOT_IDS)
        )
    )
    assignments_for(
        seeds=seed_values,
        temperature_capability=temperature_capability,
        seed_capability=seed_capability,
    )
    return SessionManifest.create(
        settings,
        session_id=session_id,
        title=title,
        seeds=seed_values,
        temperature_capability=temperature_capability,
        seed_capability=seed_capability,
        studio_version=studio_version,
        now=now,
    )


# Names used by callers that describe the operation as assignment rather than
# manifest creation.  Keeping these aliases small makes the policy convenient
# to use without introducing a second implementation.
build_assignments = assignments_for
build_manifest = create_manifest


__all__ = [
    "SLOT_IDS",
    "TEMPERATURES",
    "VariantAssignment",
    "assignments_for",
    "build_assignments",
    "build_manifest",
    "create_manifest",
]
