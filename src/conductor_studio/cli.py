"""Command-line interface for Conductor Studio."""

from __future__ import annotations

import argparse
import logging
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
    if config.allow_network and config.host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        logging.warning(
            "Conductor Studio has no authentication. Do not expose this server "
            "to the public internet."
        )

    from conductor_studio.app import create_app

    app = create_app()
    app.launch(
        server_name=config.host,
        server_port=config.port,
        share=False,
        show_api=False,
    )
