"""Small helper around Text-Fabric's core API for datasets stored under data/raw."""
from __future__ import annotations

from pathlib import Path


def load_tf(location: str | Path, features: str):
    """Load a Text-Fabric dataset from a directory of .tf files and return its api (F, L, T, E ...)."""
    from tf.fabric import Fabric

    TF = Fabric(locations=[str(location)], silent="deep")
    api = TF.load(features, silent="deep")
    if not api:
        raise RuntimeError(f"Text-Fabric could not load {location} with features {features!r}")
    return api
