# Conductor Studio

Conductor Studio is a local-first Gradio application for comparing four musical
loop variants generated through Conductor Core. One session owns an immutable
prompt, key, scale, provider, model, and reasoning selection. The four cards
share those settings while using the fixed Studio temperature ladder `0.2`,
`0.3`, `0.4`, and `0.5` whenever the selected Core adapter can make that value
effective.

The filesystem under `~/.conductor/studio/` is the source of truth. Studio is a
personal, single-process tool: it has no accounts, authentication, billing,
public sharing, or hosted multi-user mode.

## Prerequisites

- Windows, macOS, or Linux with Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) 0.12.6 or newer.
- FluidSynth and FFmpeg on `PATH` for optional MP3 previews. MIDI generation
  and downloads do not require the audio toolchain.
- Provider credentials only for the providers you intend to use. Ollama also
  requires a reachable local Ollama service.

## Install and run

Run these commands from the Studio repository root. No shell activation is
needed; uv creates and uses `.venv` automatically.

```powershell
uv sync --all-groups
uv run conductor-studio
```

The default server binds to `127.0.0.1` and Gradio sharing is always disabled.
To deliberately bind to another interface, opt in explicitly:

```powershell
uv run conductor-studio --host 0.0.0.0 --port 7860 --allow-network
```

This prints a prominent no-authentication warning. Do not expose Studio to the
public internet or treat `--allow-network` as an access-control feature.

Studio installs Core from the pinned remote revision below; it never imports a
sibling checkout:

```text
https://github.com/laceyp99/conductor-core.git
068b85d7471602a2cf18476d5207c466f088bf41
```

## Credentials and provider readiness

Core's documented environment variables are:

| Provider | Environment variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Ollama host | `OLLAMA_API_HOST_ADDRESS` |

The Settings workflow accepts temporary values in process memory. A session
override takes precedence over its environment value; clearing the override
restores environment fallback. Values are never written to manifests, logs, or
the filesystem, and the UI reports only a source/configured status. Restarting
the process clears overrides.

Cloud readiness means that a credential is configured; Studio does not make a
billable validation request just to turn a status green. Ollama discovery is a
non-billable status request with a short timeout and an explicit refresh. Its
model names are live and are not part of Core's packaged cloud catalog.

Provider and model are separate controls. The catalog is derived from Core's
validated metadata. Thinking and effort controls appear only when the selected
model exposes those capabilities. Temperature behavior is provider/model
dependent, and the current pinned Core revision has no seed request field or
seed capability metadata: Studio must record seed support as unavailable and
must not emulate a seed by changing the prompt.

## MVP workflow

1. Open the Generate tab and enter a nonblank prompt, key, scale, provider,
   model, and applicable reasoning settings.
2. Submit once. Studio creates a new session and persists its immutable
   settings plus four queued slots before any provider call.
3. Watch the fixed four-card grid update independently through provider, MIDI,
   and audio stages. A successful MIDI remains available if audio cannot
   render.
4. Download MIDI/audio artifacts, favorite variants, or retry only failed or
   interrupted slots. Successful siblings are not regenerated.
5. Reopen completed work from History or Favorites without contacting a
   provider. Delete moves a complete terminal session to managed Trash rather
   than permanently removing it.

## Audio behavior

Setups with FluidSynth, FFmpeg, and Core's packaged SoundFont can render an MP3
preview. Missing tools, SoundFont discovery failures, and render errors are
non-fatal: Studio preserves the generated MIDI, reports an actionable audio
status, and allows an audio-only retry where supported. Audio retry must never
make another LLM request.

## Data, recovery, and backups

By default, Studio uses:

```text
~/.conductor/studio/
  sessions/
    YYYYMMDD-HHMMSS_shortuuid/
      session.json
      session.previous.json
      variants/01/ ... variants/04/
        core/                 # Core artifacts, including loop.mid/messages/metadata
        piano-roll.png
  served/                     # only directory allowed for Gradio file serving
  trash/                      # recoverable deleted session folders
```

Set `CONDUCTOR_HOME` to move the suite root. Studio does not honor Core's
`CONDUCTOR_CORE_DATA_DIR` for its own manifests. Each manifest is versioned,
validated, and replaced atomically; the prior valid manifest is retained as
`session.previous.json`. Completed slot results are persisted immediately.

On startup, queued or active slots are marked `interrupted`; Studio never
automatically retries an uncertain provider call. Retry them manually after
checking credentials and provider readiness. If a manifest becomes unreadable,
keep the session directory intact and inspect the previous manifest. Trash is
under the managed root and can be manually restored by moving a complete
session directory back beneath `sessions/`; do not copy only individual files.
Back up the complete `sessions/` and `trash/` directories while Studio is
stopped or between writes.

## Security boundary

The default localhost bind is intentional. Studio has no authentication and
can issue provider requests using a user's credentials, so LAN binding is an
explicit opt-in with a warning. Gradio `share` remains disabled; public links,
reverse-proxy deployment, and public hosting are unsupported.

Only the dedicated `served/` directory is allowed for Gradio file serving;
session and trash roots are blocked. Do not place credentials, arbitrary local
files, or generated secrets in served content.

## Developer checks

Use the locked environment and run from the repository root:

```powershell
uv sync --all-groups
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest -q
uv build
```

Tests are deterministic and should not make live provider calls. Keep Core
behind `conductor_studio.core_adapter`; do not import provider SDKs directly
from Studio. Leave `plan.md` and review/scratch markdown uncommitted.

## Manual verification checklist

The following checks require a local environment and are intentionally not
claimed by the automated suite until run on that environment:

- [ ] Launch offline on `127.0.0.1`; confirm the app loads without credentials.
- [ ] Configure one environment credential and verify source status without
      revealing its value.
- [ ] Set, clear, and restart a session override; verify environment fallback
      and process-memory lifetime.
- [ ] Refresh Ollama against both a reachable and unreachable host.
- [ ] Generate four variants and confirm stable progressive four-card layout.
- [ ] Force one slot failure; verify three successes remain and only the failed
      slot is retried.
- [ ] Remove FluidSynth or FFmpeg; verify MIDI remains usable and audio status
      is actionable; restore the tool and retry audio only.
- [ ] Terminate during generation; restart and verify active slots are
      interrupted rather than automatically retried.
- [ ] Reopen History/Favorites without network activity.
- [ ] Favorite variants, move a terminal session to Trash, and manually
      restore its complete folder.
- [ ] Verify desktop 2x2 and narrow one-column layouts in a current browser.
- [ ] Launch with `--allow-network` and verify the no-auth warning while
      `share=False` remains in effect.
- [ ] When explicitly configured, run one provider smoke test for OpenAI,
      Anthropic, Gemini, and Ollama; record failures rather than treating them
      as validated.

## MVP limitations

Studio intentionally does not provide prompt refinement/branching, variable
variant counts, manual temperature or seed controls, automatic retries, true
cancellation, synchronized playback cursors, in-app MIDI editing, user
SoundFont management, search/tags/import/export, permanent deletion, accounts,
authentication, public sessions, billing, or hosted multi-user isolation.

Seed reproducibility is unavailable with the pinned Core contract. Temperature
variation is a requested Studio policy and may not be effective for reasoning
or provider models whose adapters omit or override it.
