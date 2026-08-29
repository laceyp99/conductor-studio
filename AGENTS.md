# Conductor Studio contributor guidance

- Keep Conductor Core behind `core_adapter.py`; Studio must not call provider SDKs.
- Never persist credentials, provider request payloads, or arbitrary local paths.
- Keep session manifests authoritative under the configured Studio root.
- Every session has exactly four slots and immutable generation settings.
- Serialize manifest mutations and use atomic replacement with a validated backup.
- Do not automatically retry interrupted or failed provider calls.
- Treat MIDI success independently from optional audio rendering.
- Keep Gradio sharing disabled and serve only contained Studio artifacts.
- Run the locked Ruff, pytest, and build gates before publishing changes.
- Keep `plan.md` and review/scratch markdown files uncommitted.
