"""Gradio application composition."""

from __future__ import annotations


def create_app():
    """Create the application lazily so package imports stay offline-safe."""
    import gradio as gr

    with gr.Blocks(title="Conductor Studio") as app:
        gr.Markdown("# Conductor Studio\n\nThe four-variant studio is initializing.")
    return app
