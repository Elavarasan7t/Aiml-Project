"""Root proxy for backend.main:app to allow running uvicorn from workspace root."""
import sys
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
inner_backend = proj_root / "backend"

for p in [str(proj_root), str(inner_backend)]:
    if p not in sys.path:
        sys.path.insert(0, p)

real_main_path = inner_backend / "main.py"
with open(real_main_path, "r", encoding="utf-8") as f:
    _code = compile(f.read(), str(real_main_path), "exec")
exec(_code, globals())
