"""Shared config/paths helpers used by every pipeline script."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "project_config.yaml") as f:
        return yaml.safe_load(f)


def load_blocks() -> list:
    with open(PROJECT_ROOT / "configs" / "blocks.yaml") as f:
        return yaml.safe_load(f)["blocks"]


def get_block(block_id: str) -> dict:
    for b in load_blocks():
        if b["block_id"] == block_id:
            return b
    raise KeyError(f"Block '{block_id}' not found in configs/blocks.yaml")


def resolve_path(p) -> Path:
    """Absolute paths are used as-is; relative paths (config or block data) resolve against PROJECT_ROOT."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
