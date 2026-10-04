# Conductor Studio

Conductor Studio is a local-first Gradio application for comparing four musical
loop variations. One Generate click creates a new Studio session and makes one
Conductor Core batch request for four ordered items. The prompt, musical
settings, provider, model, and generation controls are immutable for that
session; generate again to start a separate request.

Studio is a personal, single-process tool. It has no accounts, authentication,
billing, public sharing, or hosted multi-user mode.

## Prerequisites

- Windows, macOS, or Linux with Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) 0.12.6 or newer.
- FluidSynth and FFmpeg on `PATH` for optional MP3 previews. MIDI generation
  and downloads do not require the audio toolchain.
- Credentials for the providers you use. Ollama also needs a reachable service.

## Install and run

```powershell
uv sync --all-groups
uv run conductor-studio
```

The server binds to `127.0.0.1` and Gradio sharing is disabled. Network binding
is an explicit, unauthenticated opt-in:

```powershell
uv run conductor-studio --host 0.0.0.0 --port 7860 --allow-network
```

Studio installs Conductor Core v0.6.0 from the immutable peeled commit below,
never from a sibling checkout or mutable branch:

```text
https://github.com/laceyp99/conductor-core.git
11f18591e4bb788180ca765f753d1b62b2380251
```

## Credentials and controls

| Provider | Environment variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Ollama host | `OLLAMA_API_HOST_ADDRESS` |

Settings can hold temporary overrides in process memory. Overrides take
precedence over the environment and disappear at restart; credentials and raw
provider messages are never persisted.

Provider and model controls come from Core metadata, never from provider or
model names. Each model gets at most one way to choose reasoning:

- **Reasoning effort** for every model with effort levels. When reasoning can
  be turned off (`thinking_off: "disabled"`), the lowest level is `none`:
  either Core's own `none` effort (OpenAI GPT-5.1 and later), or one Studio
  adds that sends Core's thinking-off setting (Claude Opus 4.6 through Opus 5,
  Sonnet 4.6 and 5, and Ollama models that accept `think=false`). Models whose
  reasoning cannot be turned off (`thinking_off: "lowest_effort"`) get no
  `none` level.
- **Extended thinking** toggle for models without effort levels that can turn
  reasoning off, such as the Claude 4.5 models and Gemini 2.5 Flash.
- No reasoning control when reasoning cannot be turned off and there are no
  levels (for example Gemini 2.5 Pro). Studio requests thinking, and Core uses
  the model's full thinking budget.
- **Temperature** appears only when `temperature_supported` is not `false`. While
  thinking, a model's `thinking_fixed_temperature` (for example `1.0` for
  Claude Opus 4.6, Sonnet 4.6, and the Claude 4.5 models) locks the slider and
  becomes the effective temperature. Other models keep the requested value,
  which is restored when thinking is turned off or `none` is chosen.

Studio opens on Google's newest model (the first one Core lists) with the
description "a rhythmic sad pop piano". The default requested temperature is `0.7`
within Studio's `0.0`–`1.0` slider range for all models, avoiding values that
some providers reject. The pinned Core contract has no seed field, so
Studio neither offers nor emulates one.

Ollama models are discovered when the page loads and again on **Refresh Ollama
models**. Their controls follow the same rules, based on the per-model
capabilities Core reads from Ollama. Ollama reports no fixed thinking
temperature, so the slider stays adjustable while thinking. For Ollama only,
a collapsed **Advanced Settings** section holds **Ollama Context Size**:
**Ollama default** (the default, which sends no `num_ctx`, so the server's
`OLLAMA_CONTEXT_LENGTH` or Modelfile applies) or 1,024, 4,096, 16,384, 65,536,
or 262,144 tokens. The choice is saved with the session, sent as Core's
`ollama_num_ctx`, and shown with the shared session settings above the cards. Larger
windows use more memory and can stop a model from loading. Core does not report
a model's maximum context, so Studio does not hide larger presets. If a batch
fails because the model ran out of context, lower or turn off thinking, choose
a model with a larger context, or pick a larger preset. Studio never retries
automatically.

## Batch workflow and accounting

1. In the left column choose a key and scale and enter a description; in the
   right column choose a provider, model, and the controls offered for that
   model. Then select **Generate Variations**.
2. Submit once. Studio first persists the complete queued manifest, then sends
   one request for four variations.
3. All four cards share provider progress. Ordered MIDI processing may be shown,
   but no result reference or download is published until all four items pass
   validation and are saved atomically.
4. Cost and available input, output, and total token counts are recorded once
   for the batch and displayed beneath its cards.
5. A provider, validation, or persistence error fails the entire batch with one
   sanitized error. Start a new session to try generation again; Studio never
   repeats the provider request within an existing session.
6. Reopen completed work from History or Favorites without provider activity.
   Results appear under Generate and History; opening a favorite switches to
   Generate.

After successful MIDI publication, Studio derives loop JSON and piano-roll
images and may render up to four MP3 previews. Those jobs are optional: a roll
or audio failure never invalidates MIDI. Audio-only retry uses the existing
contained MIDI and never makes an LLM request.

## Data, recovery, and Trash

The filesystem under `~/.conductor/studio/` is authoritative:

```text
~/.conductor/studio/
  sessions/
    YYYYMMDD-HHMMSS_shortuuid/
      session.json
      session.previous.json
      core/
        generations/          # Core child records and MIDI
        variations/           # Core batch records
      variants/01/ ... variants/04/
        loop.json
        piano-roll.png
        preview.mp3
  served/                     # copied, approved media only
  trash/                      # complete recoverable session folders
```

`CONDUCTOR_HOME` moves the suite root. Every manifest mutation is validated,
serialized, and atomically replaced; the preceding valid manifest is retained
as `session.previous.json`.

At startup, a nonterminal batch is marked interrupted with a sanitized failure,
including any in-progress automatic audio. Studio does not inspect Core history,
reconstruct results, or automatically retry. A rare crash after Core completes
but before Studio publishes the batch can therefore leave unreferenced Core
records under that session; a later new session may incur another provider
charge.

Deleting a terminal session moves its whole directory to managed Trash in one
operation. Its manifests, backups, `core/generations`, `core/variations`, and
derived variants stay together. Restore by moving the complete directory back;
do not copy individual Core files.

## Security boundary

Only regular, contained media is copied into `served/`: Core generation MIDI
(`.mid`/`.midi`) and Studio-derived audio or images (`.mp3`/`.png`). Session
manifests, Core metadata, variation records, provider messages, loop JSON, and
anything beneath Trash are never served directly. Do not expose the
unauthenticated network mode to the public internet.

## Developer checks

```powershell
uv sync --all-groups
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest -q
uv build
```

Tests are offline and deterministic. Keep Core behind
`conductor_studio.core_adapter`; Studio must not import provider SDKs directly.
Leave `plan.md`, `decisions.md`, and review/scratch notes uncommitted.

## Manual verification

- Launch on localhost and verify all three capability-driven control modes.
- Generate once and confirm shared request progress, ordered MIDI processing,
  atomic four-download publication, and batch accounting.
- Force a batch failure and verify all cards fail together with one sanitized
  error; Generate should create a new session.
- Remove FluidSynth or FFmpeg and verify MIDI remains usable, then retry audio
  without provider activity.
- Terminate during generation, restart, and verify interruption without Core
  reconstruction or retry.
- Reopen History/Favorites, move a terminal session to Trash, and confirm the
  complete Core and variant trees moved while `served/` contains media only.

Live OpenAI, Anthropic, Gemini, and Ollama checks are opt-in because they require
credentials and can incur cost. When not explicitly run, their status is
unverified rather than passed or failed.

## MVP limitations

Studio intentionally omits variable batch sizes, provider-generation retry,
automatic recovery, cancellation, in-app MIDI editing, user SoundFont
management, search/tags/import/export, permanent deletion, accounts,
authentication, public sessions, billing, and hosted multi-user isolation.
