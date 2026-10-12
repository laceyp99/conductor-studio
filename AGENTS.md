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

- Never persist credentials, provider request payloads, or arbitrary local
  paths. Credential overrides are process-memory only.
- Keep session manifests authoritative under the configured Studio root.
- Serve files only from the dedicated `served/` directory; block `sessions/`
  and `trash/` from Gradio file serving.
- Keep Gradio sharing disabled. The default bind is localhost; any non-loopback
  bind requires explicit `--allow-network` and a no-authentication warning.

## Browser verification over LAN and Tailscale

- Identify where Studio, the browser, and any Ollama service run.
  `localhost` refers to the machine making the connection. The Ollama
  host must be reachable from Studio, independently of browser access.

- Verify Studio responds on its server host before testing remotely.
  For a browser on another device, use the server's reachable LAN or
  Tailscale address. Prefer Tailscale for private remote previews.

- Use Studio's supported `--host`, `--port`, and `--allow-network`
  options. Bind to the Tailscale interface for Tailscale-only access;
  use `0.0.0.0` when both LAN and Tailscale access are intended.
  Open the server's actual address in the browser.

- Preserve `share=False` and the existing file-serving restrictions.
  Disclose that Studio has no authentication when enabling network access.

- Prefer the environment's native browser tools and follow their
  documented fallback rules. If automatic preview routing fails, try
  a verified direct server URL.

- Diagnose startup, binding, address, routing, host permission prompts,
  and browser-tool failures separately. Do not change firewall rules,
  access policies, or enable public sharing without authorization.

- Report which addresses and devices were verified, temporary
  workarounds, and remaining gaps. Stop temporary servers after
  verification unless the user needs them for review.

## Domain and persistence

- Every session has exactly four slots and immutable generation settings.
- Serialize manifest mutations and use atomic replacement with a validated
  backup.
- Do not automatically retry interrupted or failed provider calls.
- Treat MIDI success independently from optional audio rendering. Audio retry
  must not issue another LLM request.
- Move deleted terminal sessions to managed Trash; do not permanently delete
  creative artifacts in the MVP.

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
