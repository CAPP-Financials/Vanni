"""Resolve app paths in both source and PyInstaller-frozen runs."""
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)

# editable files (config.toml, corrections.json) live next to the exe when frozen
BASE = Path(sys.executable).parent if FROZEN else Path(__file__).parent

# bundled read-only resources (DLLs collected into the bundle)
BUNDLE = Path(getattr(sys, "_MEIPASS", BASE))
