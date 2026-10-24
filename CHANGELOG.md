# Changelog

## Unreleased

### Changed

- Session artifacts are served directly from `studio/sessions/<session-id>/`.
  File access is limited to sessions; Gradio's upload/media cache stays blocked.
- Deleting a terminal session permanently removes its artifacts, favorites,
  manifests, and Core records. Generation and audio rendering prevent deletion.
- Failed deletion shows an inline retry banner and retains the selection.
  If another tab already deleted the displayed session, a delete attempt clears
  its stale cards and refreshes History.

### Removed

- Trash APIs and separate media publication copies.
- Explicit `served/` and `trash/` path blocks and their compatibility assertions.
  Those directories are outside the allowed sessions path and remain inaccessible.

### Migration for upstream callers

- Replace `StudioService.move_to_trash(session_id)` and
  `SessionStore.move_to_trash(session_id)` calls with `delete_session(session_id)`.
  Deletion is permanent; callers must not offer restore behavior.
- Remove `in_trash` arguments from `SessionStore.session_dir`, `load`, and
  `artifact_path`. Remove references to `SessionStore.trash_root`.
- Replace `MediaPublisher` usage and saved publication URLs with current artifact
  paths obtained through `SessionStore.artifact_path`. Reopen sessions to obtain
  direct media URLs, and obtain a fresh audio URL after a successful retry.
- Stop Studio before moving any sessions retained in an old `trash/` directory
  into `sessions/`. Preserve each complete session folder and its manifest ID;
  resolve destination conflicts before moving. Existing sessions already beneath
  `sessions/` need no relocation. Old `served/` copies are not session records.
- After verifying retained sessions, archive or delete unused `trash/` and
  `served/` contents yourself. No automatic migration or cleanup runs. Restart
  Studio to apply the current serving policy; already downloaded media remains
  outside Studio's control.
