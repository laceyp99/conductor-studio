import os
from pathlib import Path

import pytest


@pytest.fixture
def link_dir():
    """Create a directory link (a junction on Windows) or skip if disallowed."""

    def create(link: Path, target: Path) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                import _winapi

                _winapi.CreateJunction(str(target), str(link))
            else:
                link.symlink_to(target, target_is_directory=True)
        except (OSError, AttributeError, ImportError):
            pytest.skip("directory links are unavailable")

    return create
