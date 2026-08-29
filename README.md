# Conductor Studio

Conductor Studio is a local-first Gradio application for comparing four musical
loop variants generated through Conductor Core. A session keeps one immutable
prompt and musical setup while each card receives a fixed temperature from
`0.2` through `0.5` when the selected Core model supports it.

The project is under active MVP construction. It stores sessions beneath
`~/.conductor/studio/` by default, never persists provider credentials, binds to
`127.0.0.1`, and does not enable Gradio sharing.

## Development

```powershell
uv sync --all-groups
uv run conductor-studio
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv build
```

Conductor Core is installed from the immutable Git revision recorded in
`pyproject.toml` and `uv.lock`; the sibling `conductor-core` checkout is not used.
