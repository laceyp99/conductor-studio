"""Command-line interface for Conductor Studio."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from conductor_studio.config import LaunchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Conductor Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="permit a non-loopback bind; there is no authentication",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = LaunchConfig(args.host, args.port, args.allow_network)
    if not config.is_loopback:
        warning = (
            "WARNING: Conductor Studio has no authentication and is bound to a "
            "non-loopback address; keep it on a trusted network."
        )
        # Keep this visible even when the host application has replaced the
        # root logger or configured logging to discard warnings.
        print(warning, file=sys.stderr)
        logging.getLogger("conductor_studio").warning(warning)

    from conductor_studio.app import create_app

    # Blocks starts its analytics/version-check thread in ``__init__``.  Set
    # the documented environment switch before construction so no telemetry
    # thread starts, then restore a caller-provided value after construction.
    previous_analytics = os.environ.get("GRADIO_ANALYTICS_ENABLED")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    try:
        app = create_app()
    finally:
        if previous_analytics is None:
            os.environ.pop("GRADIO_ANALYTICS_ENABLED", None)
        else:
            os.environ["GRADIO_ANALYTICS_ENABLED"] = previous_analytics
    # Gradio 6 exposes analytics on Blocks construction rather than as a
    # launch kwarg.  The current app factory is intentionally lazy, so set the
    # flag before launch and also disable monitoring in the supported launch API.
    app.analytics_enabled = False
    served_root = config.ensure_served_root()
    studio_root = served_root.parent
    app.launch(
        server_name=config.host,
        server_port=config.port,
        share=False,
        enable_monitoring=False,
        allowed_paths=[str(served_root)],
        blocked_paths=[str(studio_root / "sessions"), str(studio_root / "trash")],
    )
