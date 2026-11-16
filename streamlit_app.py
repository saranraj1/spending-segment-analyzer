"""Root entrypoint for Streamlit Community Cloud.

Redirects directly to app/main.py so Streamlit Cloud auto-detects the app without manual path configuration.
"""
import sys
from pathlib import Path
import runpy

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute app/main.py in the main execution namespace
runpy.run_path(str(ROOT_DIR / "app" / "main.py"), run_name="__main__")
