"""Load the YAML config into a plain nested dict with attribute-style access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class Config(dict):
    """A dict that also allows ``cfg.section.key`` access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - surfaced as AttributeError
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from ``path``, the ``CIA_CONFIG`` env var, or the default file."""
    config_path = Path(path or os.environ.get("CIA_CONFIG", DEFAULT_CONFIG))
    with open(config_path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
