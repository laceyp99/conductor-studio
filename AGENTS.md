# Conductor Studio contributor guidance

## Boundaries

- Keep Conductor Core behind `core_adapter.py`; Studio must not call provider
  SDKs directly.
- Studio depends on the immutable Core Git revision documented in README and
  `pyproject.toml`. Never switch the dependency to a sibling checkout or a
  mutable branch for release work.
- Keep model/provider controls capability-driven from Core metadata. Do not
  hard-code provider prefixes, model lists, seed support, or temperature
  behavior.
- Exception: provider-specific behavior is allowed only when the maintainer has
  approved it and Core metadata cannot express it.
- Studio relies on Core's documented reasoning vocabulary: `thinking_off`
  values and the `none` effort that turns reasoning off. For models with effort
  levels and `thinking_off: "disabled"` but no `none` level, Studio adds `none`
  to the dropdown and sends it as `use_thinking=False`, never as an effort.
- The current pinned Core contract has no seed field. Never emulate a seed by
  mutating prompts or claim reproducibility that Core cannot provide.

## Secrets and local serving

- Never persist credentials or arbitrary local paths. Credential overrides are
  process-memory only.
- Core stores each batch's provider messages (system, user, reasoning, and
  output) in its batch record under the session. Keep them; do not strip or
  copy them elsewhere, and never log them.
- Keep session manifests authoritative under the configured Studio root.
- Serve artifacts directly from `sessions/` using Gradio static paths, without
  publication copies. This intentionally exposes session manifests, metadata,
  and stored provider messages to clients of the private app. Limit file serving to `sessions/` and block
  Gradio cache paths. Never persist credentials in session records.
- Keep Gradio sharing disabled and default to loopback.
  For authorized remote previews, bind to the server host's Tailscale IP using
  `--host <tailscale-ip> --allow-network`. Keep the no-authentication warning
  visible. `--allow-network` permits binding; it does not authenticate users.

## Domain and persistence

- Every session has exactly four slots and immutable generation settings.
- Serialize manifest mutations and use atomic replacement with a validated
  backup.
- Do not automatically retry interrupted or failed provider calls.
- Treat MIDI success independently from optional audio rendering. Audio retry
  must not issue another LLM request.
- Permanently delete terminal sessions on explicit user action. Refuse deletion
  during generation or audio rendering. Keep session artifacts beneath their
  session directory.
- Core owns audio storage. Commit rendered audio through Core's staged audio
  commit (`stage_generation_audio`) and do not add Studio-side staging,
  rollback, or file-replacement logic.
- Studio's audio has not moved onto that commit yet. Until it does, keep the
  interim path: a new preview filename under `variants/` per successful render,
  with the previous file removed, so retries load fresh audio when sessions are
  served directly.

## Validation

Use uv without activating the environment:

```powershell
uv sync --all-groups
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest -q
uv build
```

Keep tests offline and deterministic. Provider/audio smoke tests are explicit,
opt-in, and must be reported as unverified when unavailable. Preserve user
changes in a dirty worktree. Keep `plan.md` and review/scratch markdown files
uncommitted unless explicitly requested.
