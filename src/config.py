"""
Configuration loading utilities.

Loads config.yaml into a nested, attribute-accessible object so the rest of
the codebase can do `cfg.training.learning_rate` instead of juggling raw
dictionaries.
"""
import os
import yaml


class ConfigDict(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[item] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str = None) -> ConfigDict:
    """Load config.yaml (or a given path) into a ConfigDict."""
    if path is None:
        # default: config.yaml at project root, two levels up from src/
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "config.yaml")
    path = os.path.abspath(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigDict(raw)


def get_project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def resolve_path(relative_path: str) -> str:
    """Resolve a path in config.yaml (relative to project root) to an absolute path."""
    root = get_project_root()
    return os.path.join(root, relative_path)
