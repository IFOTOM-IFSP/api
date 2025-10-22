from __future__ import annotations
from functools import lru_cache

from .image_processor import SpectraProcessor

@lru_cache(maxsize=1)
def get_spectra_processor() -> SpectraProcessor:
    return SpectraProcessor()
