"""Serve the canonical option inventory to the frontend."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache
def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
