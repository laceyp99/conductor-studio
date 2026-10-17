"""Safe publication of Studio media for Gradio file components."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath

from .models import ArtifactRefs, SessionManifest, VariantSlot
from .storage import ContainmentError, SessionStore

_ALLOWED_SUFFIXES = frozenset({".png", ".mp3", ".mid", ".midi"})


class MediaPublisher:
    """Copy only validated session artifacts into a Gradio-served directory.

    Session manifests and Core metadata remain outside ``served_root``.  The
    returned paths are therefore safe to pass to Gradio's file components when
    ``launch(allowed_paths=[served_root])`` is used.
    """

    def __init__(self, served_root: str | Path) -> None:
        selected = Path(served_root).expanduser().absolute()
        selected.parent.mkdir(parents=True, exist_ok=True)
        if selected.exists() or selected.is_symlink():
            info = selected.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise ContainmentError("served root cannot be a link or reparse point")
            if not selected.is_dir():
                raise ContainmentError("served root must be a directory")
        else:
            selected.mkdir()
        self.served_root = selected.resolve()

    def _destination(self, session_id: str, slot_id: str, source: Path) -> Path:
        if not session_id or not slot_id or slot_id not in {"01", "02", "03", "04"}:
            raise ContainmentError("invalid media identity")
        if source.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ContainmentError("artifact type is not safe to serve")
        destination_dir = self.served_root / session_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{slot_id}-{source.name}"
        resolved = destination.resolve()
        try:
            resolved.relative_to(self.served_root)
        except ValueError as exc:
            raise ContainmentError("served artifact escaped managed root") from exc
        return resolved

    def publish_path(
        self,
        store: SessionStore,
        session_id: str,
        slot_id: str,
        relative_path: str | None,
    ) -> Path | None:
        """Validate and copy one manifest-relative artifact, or return ``None``."""
        if not relative_path:
            return None
        parsed = PurePosixPath(relative_path)
        suffix = parsed.suffix.lower()
        if suffix in {".mid", ".midi"}:
            if len(parsed.parts) < 4 or parsed.parts[:2] != (
                "core",
                "generations",
            ):
                raise ContainmentError("MIDI must be a Core generation artifact")
        elif not parsed.parts or parsed.parts[0] != "variants":
            raise ContainmentError("derived media must be stored beneath variants")
        source = store.artifact_path(session_id, relative_path)
        destination = self._destination(session_id, slot_id, source)
        with source.open("rb") as source_stream:
            info = os.fstat(source_stream.fileno())
            # Copies carry the source size and mtime, so an unchanged artifact
            # is reused while a same-path replacement is copied again.
            with suppress(OSError):
                current = destination.stat()
                if (current.st_size, current.st_mtime_ns) == (
                    info.st_size,
                    info.st_mtime_ns,
                ):
                    return destination
            # Copy to a sibling temporary file and replace so Gradio never sees
            # a partially-written media file during progressive updates.
            _descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(_descriptor)
            temporary = Path(temporary_name)
            try:
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(source_stream, stream)
                os.utime(temporary, ns=(info.st_atime_ns, info.st_mtime_ns))
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    def unpublish(self, session_id: str) -> None:
        """Delete one session's served copies, e.g. after it moves to Trash.

        Only the disposable copies go; the session's own artifacts are untouched.
        """
        if session_id in {"", ".", ".."} or Path(session_id).name != session_id:
            raise ContainmentError("invalid media identity")
        target = self.served_root / session_id
        try:
            info = target.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ContainmentError("served session cannot be a link or reparse point")
        shutil.rmtree(target)

    def publish_slot(
        self, store: SessionStore, manifest: SessionManifest, slot: VariantSlot
    ) -> dict[str, Path | None]:
        artifacts: ArtifactRefs = slot.artifacts
        published: dict[str, Path | None] = {}
        for kind, relative_path in (
            ("piano_roll", artifacts.piano_roll),
            ("audio", artifacts.audio),
            ("midi", artifacts.midi),
        ):
            # One unavailable artifact must not hide the other usable media.
            # Files can disappear between containment validation and copying.
            with suppress(ContainmentError, OSError):
                published[kind] = self.publish_path(
                    store, manifest.session_id, slot.slot_id, relative_path
                )
            published.setdefault(kind, None)
        return published


__all__ = ["MediaPublisher"]
