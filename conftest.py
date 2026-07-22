import sys
from pathlib import Path

# Make `src` and `scripts` importable when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
