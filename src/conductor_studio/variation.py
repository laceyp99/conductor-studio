"""Construction policy for one immutable four-item batch manifest."""

from .models import SessionManifest, SessionSettings, VariantSlot

SLOT_IDS = VariantSlot.SLOT_IDS


def create_manifest(
    settings: SessionSettings,
    *,
    core_version: str,
    session_id=None,
    title=None,
    studio_version="0.1.0",
    now=None,
) -> SessionManifest:
    return SessionManifest.create(
        settings,
        core_version=core_version,
        session_id=session_id,
        title=title,
        studio_version=studio_version,
        now=now,
    )


build_manifest = create_manifest
__all__ = ["SLOT_IDS", "build_manifest", "create_manifest"]
