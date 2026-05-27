from __future__ import annotations

import sys
from pathlib import Path

# Ensure test imports work consistently in local and CI runners.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
