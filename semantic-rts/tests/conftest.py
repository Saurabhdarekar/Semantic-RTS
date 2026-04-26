"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from semantic_rts.config import Config, load_config


@pytest.fixture
def default_config() -> Config:
    """Load the default config (no API calls needed)."""
    return load_config()


@pytest.fixture
def no_bridge_config() -> Config:
    """Default config with Safety Bridge disabled."""
    return load_config(overlay_path="config/ablations/no_bridge.yaml")
