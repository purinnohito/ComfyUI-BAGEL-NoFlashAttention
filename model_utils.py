"""Utility functions for BAGEL model discovery and detection.

This module centralizes helpers used by nodes.py to keep that file concise.
"""
import os
from typing import Dict

from folder_paths import models_dir as comfy_models_dir


def discover_bagel_model_dirs() -> Dict[str, str]:
    """Discover local BAGEL model directories under the configured models/bagel folder.

    Returns a mapping from folder-name -> absolute-path.
    """
    base_repo_dir = os.path.join(comfy_models_dir, "bagel")
    discovered: Dict[str, str] = {}
    try:
        if os.path.exists(base_repo_dir):
            for name in sorted(os.listdir(base_repo_dir)):
                p = os.path.join(base_repo_dir, name)
                if os.path.isdir(p):
                    discovered[name] = p
    except Exception as e:
        print(f"Error discovering bagel model dirs: {e}")
    return discovered


def is_df11_name(name: str) -> bool:
    """Heuristic to detect DFloat11 derived model names.

    Matches folder or repo names containing dfloat11
    """
    if not name:
        return False
    low = name.lower()
    kws = ["dfloat11", "df11", "df11-", "df11_"]
    return any(k in low for k in kws)
