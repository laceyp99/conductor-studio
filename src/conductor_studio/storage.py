"""Crash-safe filesystem persistence for Studio sessions.

The filesystem is the source of truth.  A process-local lock serializes all
mutations to a session; the MVP intentionally does not pretend to coordinate
multiple Studio processes writing the same data root.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from .config import resolve_studio_root
from .models import (
    AudioState,
    ErrorCategory,
    FailureInfo,
    MidiState,
    SessionManifest,
    SlotState,
)

MANIFEST_NAME = "session.json"
PREVIOUS_MANIFEST_NAME = "session.previous.json"
_REPARSE_POINT = 0x400


class StorageError(RuntimeError):
    """Base class for invalid or unavailable Studio storage operations."""


class ContainmentError(StorageError):
    """A path was not safely contained beneath its managed root."""


class ManifestError(StorageError):
    """A manifest was malformed or failed domain validation."""


class UnsupportedSchemaError(ManifestError):
    """A newer (or otherwise unsupported) manifest must not be rewritten."""


class SessionStore:
    """Read and mutate versioned manifests beneath a Studio data root."""

    def __init__(self, studio_root: Path | str | None = None) -> None:
        self.studio_root = (
            Path(studio_root) if studio_root else resolve_studio_root()
        ).resolve()
        self.sessions_root = self.studio_root / "sessions"
        self.trash_root = self.studio_root / "trash"
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.RLock())

    def _ensure_roots(self) -> None:
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.trash_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reject_reparse(path: Path) -> None:
        """Reject links/reparse points in a managed path's existing components."""
        current = Path(path.anchor) if path.anchor else Path()
        for index, part in enumerate(path.parts):
            if index == 0 and path.anchor:
                continue
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
            ):
                raise ContainmentError(
                    f"managed path contains a link or reparse point: {path}"
                )

    @staticmethod
    def _contained(path: Path, root: Path, *, must_exist: bool = False) -> Path:
        root = root.resolve()
        try:
            resolved = path.resolve(strict=must_exist)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ContainmentError(f"path is outside managed root: {path}") from exc
        # Check the original spelling as well as the resolved spelling.  The
        # latter alone would follow a symlink and make an escape look safe.
        # This also checks existing parents when the final path is new.
        SessionStore._reject_reparse(path)
        if must_exist:
            SessionStore._reject_reparse(resolved)
        return resolved

    @staticmethod
    def _safe_session_id(session_id: str) -> None:
        # Let the domain model remain the single source of truth for its exact
        # Windows-safe naming grammar without constructing a fake manifest.
        if not isinstance(session_id, str) or not re.fullmatch(
            r"\d{8}-\d{6}_[A-Za-z0-9-]{4,32}", session_id
        ):
            raise ContainmentError("invalid session ID")

    def _session_dir(
        self, session_id: str, *, in_trash: bool = False, must_exist: bool = False
    ) -> Path:
        self._safe_session_id(session_id)
        root = self.trash_root if in_trash else self.sessions_root
        candidate = root / session_id
        # The direct-child check is intentional: no nested user path is ever
        # accepted as a session identifier.
        if candidate.parent != root:
            raise ContainmentError("session must be a direct child of its root")
        return self._contained(candidate, root, must_exist=must_exist)

    def session_dir(
        self, session_id: str, *, in_trash: bool = False, must_exist: bool = False
    ) -> Path:
        """Return a safely contained session directory.

        This is the supported boundary for code that needs to create derived
        artifacts beneath a session; callers must still validate individual
        files with :meth:`artifact_path` before publishing them.
        """
        return self._session_dir(session_id, in_trash=in_trash, must_exist=must_exist)

    @staticmethod
    def _read_manifest(path: Path, expected_id: str) -> SessionManifest:
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            version = document.get("schema_version")
            if version != SessionManifest.SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"unsupported manifest schema: {version!r}"
                )
            manifest = SessionManifest.model_validate(document)
        except UnsupportedSchemaError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"invalid manifest: {path}") from exc
        if manifest.session_id != expected_id:
            raise ManifestError("manifest session ID does not match its folder")
        return manifest

    @staticmethod
    def _write_temp(directory: Path, data: bytes) -> Path:
        fd, name = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=directory)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return temp

    @classmethod
    def _atomic_bytes(cls, target: Path, data: bytes) -> None:
        temp = cls._write_temp(target.parent, data)
        try:
            # Same-directory replace is atomic on supported local Windows and
            # POSIX filesystems once every source handle has been closed.
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)

    def save(self, manifest: SessionManifest) -> SessionManifest:
        """Validate and atomically persist a manifest plus one valid backup."""
        candidate = SessionManifest.model_validate(manifest.model_dump())
        session_dir = self._session_dir(candidate.session_id)
        with self._locks_guard:
            lock = self._locks.setdefault(candidate.session_id, threading.RLock())
        with lock:
            self._ensure_roots()
            session_dir.mkdir(parents=True, exist_ok=True)
            self._contained(session_dir, self.sessions_root, must_exist=True)
            current = session_dir / MANIFEST_NAME
            previous = session_dir / PREVIOUS_MANIFEST_NAME
            current_bytes: bytes | None = None
            if current.exists() or current.is_symlink():
                self._reject_reparse(current)
                try:
                    existing = self._read_manifest(current, candidate.session_id)
                except UnsupportedSchemaError:
                    # Never let an older Studio overwrite a newer manifest.
                    raise
                except ManifestError:
                    existing = None
                if existing is not None:
                    current_bytes = current.read_bytes()

            # Save the old validated current before replacing it.  If this
            # operation fails, the authoritative current file remains intact.
            if current_bytes is not None:
                self._atomic_bytes(previous, current_bytes)
            self._atomic_bytes(current, candidate.json_bytes())
            return candidate

    def create(self, manifest: SessionManifest) -> SessionManifest:
        """Create a new session, refusing to overwrite an existing one."""
        self._safe_session_id(manifest.session_id)
        self._ensure_roots()
        path = self.sessions_root / manifest.session_id
        self._contained(path, self.sessions_root)
        if path.exists():
            raise StorageError(f"session already exists: {manifest.session_id}")
        path.mkdir(parents=True)
        try:
            return self.save(manifest)
        except BaseException:
            # The empty directory is safe to clean up, but never remove a
            # directory that appeared concurrently or contains user files.
            try:
                if not any(path.iterdir()):
                    path.rmdir()
            except OSError:
                pass
            raise

    def load(self, session_id: str, *, in_trash: bool = False) -> SessionManifest:
        path = self._session_dir(session_id, in_trash=in_trash, must_exist=True)
        current = path / MANIFEST_NAME
        try:
            return self._read_manifest(current, session_id)
        except UnsupportedSchemaError:
            raise
        except ManifestError:
            previous = path / PREVIOUS_MANIFEST_NAME
            try:
                return self._read_manifest(previous, session_id)
            except UnsupportedSchemaError:
                raise
            except ManifestError as previous_error:
                raise ManifestError(
                    f"no valid manifest for {session_id}; current and previous are unusable"
                ) from previous_error

    def iter_history(self) -> Iterator[SessionManifest]:
        if not self.sessions_root.exists():
            return
        for directory in self.sessions_root.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                yield self.load(directory.name)
            except ManifestError:
                continue

    def history(self) -> list[SessionManifest]:
        return sorted(
            self.iter_history(),
            key=lambda item: (item.created_at, item.session_id),
            reverse=True,
        )

    def favorites(self) -> list[tuple[SessionManifest, str]]:
        return [
            (manifest, slot.slot_id)
            for manifest in self.history()
            for slot in manifest.slots
            if slot.favorite
        ]

    def set_favorite(
        self, session_id: str, slot_id: str, favorite: bool | None = None
    ) -> SessionManifest:
        lock = self._lock_for(session_id)
        with lock:
            manifest = self.load(session_id)
            slot = manifest.slot(slot_id)
            slot.favorite = (not slot.favorite) if favorite is None else favorite
            return self.save(manifest)

    def recover_startup(self) -> list[SessionManifest]:
        """Atomically interrupt every nonterminal batch, without Core recovery."""
        recovered: list[SessionManifest] = []
        for manifest in list(self.iter_history()):
            batch_active = not manifest.terminal
            audio_active = any(
                slot.audio.state is AudioState.RENDERING for slot in manifest.slots
            )
            if not batch_active and not audio_active:
                continue
            lock = self._lock_for(manifest.session_id)
            with lock:
                current = self.load(manifest.session_id)
                for slot in current.slots:
                    if batch_active:
                        slot.state = SlotState.INTERRUPTED
                    if slot.audio.state is AudioState.RENDERING:
                        slot.audio.state = AudioState.INTERRUPTED
                        slot.audio.failure = FailureInfo(
                            category=ErrorCategory.INTERRUPTED,
                            message="Audio rendering was interrupted. MIDI is still available.",
                        )
                        slot.audio.retryable = slot.midi is MidiState.READY
                if batch_active:
                    current.batch.failure = FailureInfo(
                        category=ErrorCategory.INTERRUPTED,
                        message=(
                            "Generation was interrupted when Studio stopped. "
                            "Start a new session to generate again."
                        ),
                    )
                current.refresh_status()
                recovered.append(self.save(current))
        return recovered

    def artifact_path(
        self, session_id: str, relative_path: str, *, in_trash: bool = False
    ) -> Path:
        """Return an existing regular contained artifact for local serving."""
        if not relative_path or "\x00" in relative_path or "\\" in relative_path:
            raise ContainmentError("artifact path must be relative POSIX syntax")
        parsed = PurePosixPath(relative_path)
        if parsed.is_absolute() or any(
            part in {"", ".", ".."} for part in parsed.parts
        ):
            raise ContainmentError("artifact path contains traversal")
        if parsed.parts[0] == "core" and (
            len(parsed.parts) < 4
            or parsed.parts[:2] != ("core", "generations")
            or parsed.suffix.lower() not in {".mid", ".midi"}
        ):
            raise ContainmentError(
                "only Core generation MIDI artifacts may be referenced"
            )
        session_dir = self._session_dir(session_id, in_trash=in_trash, must_exist=True)
        candidate = session_dir.joinpath(*parsed.parts)
        result = self._contained(candidate, session_dir, must_exist=True)
        if not result.is_file():
            raise ContainmentError("artifact is not a regular file")
        return result

    def move_to_trash(self, session_id: str) -> Path:
        """Move one complete terminal session beneath the managed trash root."""
        lock = self._lock_for(session_id)
        with lock:
            source = self._session_dir(session_id, must_exist=True)
            manifest = self.load(session_id)
            if not manifest.terminal:
                raise StorageError("active sessions cannot be moved to trash")
            self._ensure_roots()
            destination = self.trash_root / session_id
            self._contained(destination, self.trash_root)
            if destination.exists():
                raise StorageError(f"trash destination already exists: {session_id}")
            # os.replace moves the directory atomically when roots share a
            # volume (the normal ~/.conductor layout), and never copies only a
            # subset of the artifacts.
            os.replace(source, destination)
            return destination


__all__ = [
    "MANIFEST_NAME",
    "PREVIOUS_MANIFEST_NAME",
    "ContainmentError",
    "ManifestError",
    "SessionStore",
    "StorageError",
    "UnsupportedSchemaError",
]
