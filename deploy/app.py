from __future__ import annotations

import os
import runpy
from pathlib import Path

# Community Cloud entrypoint. Keep the hosted process on the Supabase + Cloudflare
# path so it never imports the heavyweight local FAISS/PyTorch fallback.
os.environ.setdefault("HOSTED_LIGHTWEIGHT", "true")

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
