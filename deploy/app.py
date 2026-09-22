from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path

# Community Cloud entrypoint. Keep the hosted process on the Supabase + Cloudflare
# path so it never imports the heavyweight local FAISS/PyTorch fallback.
os.environ.setdefault("HOSTED_LIGHTWEIGHT", "true")

# Streamlit reruns in a long-lived Python process. Purge project modules so a new
# deployment cannot combine a fresh app.py with stale telecom_rag imports.
for module_name in list(sys.modules):
    if module_name == "telecom_rag" or module_name.startswith("telecom_rag."):
        del sys.modules[module_name]
importlib.invalidate_caches()

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
