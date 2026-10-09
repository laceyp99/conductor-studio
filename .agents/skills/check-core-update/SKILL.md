---
name: check-core-update
description: Use when the user asks to check for a new conductor-core version, upgrade the conductor-core dependency, migrate conductor-studio to a new Core release, or review Core release notes/changelog. Triggers include "check version of core", "update conductor-core", "migrate to core vX.Y.Z", or "did core release anything new".
---

# Check Core Update

## Boundary to preserve

`conductor-studio` is a Gradio consumer of Conductor Core. Core owns generation,
provider routing, model metadata, loop parsing, MIDI conversion, and playback.
Studio owns sessions, slots, manifests, favorites, the UI, and credential
overrides. All Core calls go through `src/conductor_studio/core_adapter.py`; the
only other direct imports are `conductor_core.music` in `catalog.py` and
`ProviderCredentials` in `credentials.py`. Do not copy Core implementation into
Studio or call provider SDKs to work around an API change. Adapt Studio to the
new public Core API, or report that the release is not compatible yet.

## Where the pin lives

Studio pins Core to an **immutable commit SHA**, not a tag name. Keep all four
in agreement:

- `pyproject.toml`: the `conductor-core[playback,providers] @ git+...@<sha>`
  dependency.
- `uv.lock`: the resolved `conductor-core` version and commit.
- `README.md`: the Core version sentence and the commit shown under it.
- `[tool.uv] constraint-dependencies` in `pyproject.toml`: Studio-side
  workarounds for Core gaps (for example `anthropic<1`). Each one should name
  the Core limitation it covers.

Never pin a sibling checkout or a mutable branch name.

## Workflow

1. Check first. Read the pinned SHA from `pyproject.toml` and confirm `uv.lock`
   and `README.md` agree. Clone Core into a scratch directory outside the repo
   and find which tag the SHA belongs to:

   ```powershell
   git ls-remote --tags https://github.com/laceyp99/conductor-core.git
   git tag --contains <pinned-sha>
   ```

   The pin may be an **untagged branch commit** (Studio has pinned unreleased
   Core fixes before). If so, find out what that commit adds beyond its base
   tag, and treat it as a feature the target release must also contain.

2. Find the newest stable tag. Compare as semantic versions and skip
   prereleases unless asked. Tags can skip versions (for example 0.6.0 to
   0.8.3), so read the changelog for versions that were never tagged. If the pin
   is already current, report that and change nothing.

3. Read the GitHub release notes (`gh release view <tag> -R
   laceyp99/conductor-core`) and the matching `CHANGELOG.md` sections for every
   version in between. Then diff the actual source between the pinned SHA and
   the target tag for the symbols Studio uses. Release notes alone do not prove
   compatibility.

4. Inventory Core usage with `rg "conductor_core" src tests` and assess at least:

   - `core_adapter.py`: `EngineConfig`, `LoopGenerationEngine.generate_variations`,
     `VariationGenerationRequest`, `VariationBatchResult` and its item and
     metadata fields, `ProgressEvent`, `ProviderRequestError` /
     `ProviderContextLengthError`, `playback`, and `__version__` (recorded in
     every session manifest as `core_version`).
   - Ollama discovery: `providers.ollama.get_model_list` and `get_model_status`.
     Studio passes `request_timeout` to `get_model_status`, and a catalog test
     asserts the real HTTP request honors it. As of v0.8.3, `get_model_list`
     takes no timeout, so name discovery is unbounded. If a newer Core adds a
     timeout there, pass `ollama_timeout` through again.
   - Model metadata used by `catalog.py`: provider and model lists,
     temperature support, `thinking_off` values, effort levels, and the `none`
     effort. Per `AGENTS.md`, controls must stay metadata-driven. New models
     should appear without Studio code changes; confirm that rather than adding
     names.
   - Tests that encode the contract: `tests/test_compatibility.py`
     (field lists and signatures), `tests/test_core_adapter.py` (including the
     expected `core_version`), and `tests/test_catalog.py`.

   For each release, state whether it is drop-in, needs a Studio change, or is
   blocked. Also check whether any `constraint-dependencies` workaround can be
   removed because Core fixed the underlying gap.

5. Stop after the check and impact report unless the user asked you to update,
   upgrade, or migrate. A version check does not authorize edits.

6. When updating, change the SHA in `pyproject.toml` to the target tag's commit
   (`git rev-parse <tag>^{commit}`), then run `uv lock` so it resolves that exact
   commit. Do not hand-edit `uv.lock`. Update the README pin text. Make only the
   changes the new public API requires, and update tests that pin
   version-specific values. If a feature the old pin had is missing from the
   target tag, do not silently drop it or rebuild it in Studio; report the
   upstream fix needed and let the user choose.

7. Validate offline, without provider calls:

   ```powershell
   uv sync --all-groups
   uv run --locked ruff format --check .
   uv run --locked ruff check .
   uv run --locked pytest -q
   uv build
   ```

   Then launch the app locally (`uv run conductor-studio`, localhost only) and
   confirm the provider and model dropdowns populate from the new metadata.
   Real generation and Ollama/audio smoke tests are opt-in; report them as
   unverified unless the user approves and they are available.

## Report

Report the current pin and its tag, the target release and commit, the releases
inspected, and the relevant API or behavior changes. When updating, also report
files changed, the new lockfile commit, any removed or kept workarounds, and
every validation result. If blocked, name the missing Core contract and the
smallest upstream or Studio-side fix.

## Safety

- Never make paid or live provider calls without explicit approval.
- Never modify or delete existing sessions, Trash, or served files.
- Existing session manifests record the old `core_version`; a migration must
  keep them readable.
- Do not commit credentials, generated sessions, build output, `plan.md`, or
  the scratch Core clone.
