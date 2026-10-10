---
name: check-core-update
description: Use when the user asks to migrate, update, upgrade, or bump conductor-studio's conductor-core dependency to the latest (or a named) Core release.
---

# Check Core Update

## Boundary to preserve

`conductor-studio` is a Gradio consumer of Conductor Core. Core owns generation,
provider routing, model metadata, MIDI conversion, and playback. Studio owns
sessions, the UI, and credential overrides. Studio reaches Core through its
adapter layer. Do not copy Core implementation into Studio or call provider SDKs
to work around an API change. Adapt Studio to the new public Core API, or report
that the release is not compatible yet.

## The pin

Studio pins Core to an **immutable commit SHA**, never a sibling checkout or a
mutable branch. The dependency declaration, the lockfile, and the README must
agree on that commit. Any Studio-side dependency constraint that works around a
Core gap should name the gap it covers.

## Workflow

1. Identify the current pin. Merged Core PRs are tagged and released
   automatically, so expect the pin to match a release tag exactly
   (`git tag --points-at <pinned-sha>` in a scratch clone outside the repo).
   If it does not, the pin is an untagged branch commit: find what it adds over
   the nearest release and confirm the target release still contains it.

2. Find the newest stable release. If the pin is already current, report that
   and change nothing.

3. Read the release notes and changelog for every version in between, then diff
   the Core source for the APIs Studio uses. Release notes alone do not prove
   compatibility.

4. Assess impact on what Studio depends on Core for:

   - Generating a batch of variations, with progress, results, cost, and
     provider errors.
   - Model metadata that drives the provider, model, temperature, and reasoning
     controls. Controls must stay metadata-driven; new models should appear
     without Studio code changes.
   - Local Ollama discovery and model inspection, including request bounds.
   - MIDI playback and audio rendering.
   - The Core version recorded in session manifests.
   - Tests that encode the Core contract.

   Also check whether any Studio-side workaround can be removed because Core
   fixed the underlying gap.

5. Migrate by default whenever a newer stable release exists. Stop with the
   impact report only when the target needs a public Core API that is missing,
   or when migrating would drop a feature the current pin has. In that case,
   report the upstream fix needed and ask the user how to proceed; do not
   silently drop the feature or rebuild it in Studio.

6. To migrate, point the dependency at the target tag's commit and regenerate
   the lockfile with uv (never hand-edit it). Update the README pin. Make only
   the changes the new public API requires, and update tests that pin
   version-specific values.

7. Run the offline validation in `AGENTS.md`, then launch the app on localhost
   and confirm the provider and model controls populate from the new metadata.
   Live generation, Ollama, and audio smoke tests are opt-in; report them as
   unverified unless the user approves and they are available.

## Report

Report the previous and new pin with their tags, the releases inspected, and
the relevant API or behavior changes. When migrating, also report files
changed, removed or kept workarounds, and every validation result. If blocked,
name the missing Core contract and the smallest upstream or Studio-side fix.

## Safety

- Never make paid or live provider calls without explicit approval.
- Never modify or delete existing sessions, Trash, or served files.
- Existing session manifests record the old Core version; a migration must
  keep them readable.
- Do not commit credentials, generated sessions, build output, `plan.md`, or
  the scratch Core clone.
