"""Backend proxy package for workspace root."""
import sys
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
inner_backend = proj_root / "backend"

if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

if inner_backend.exists() and str(inner_backend) not in __path__:
    __path__.append(str(inner_backend))
