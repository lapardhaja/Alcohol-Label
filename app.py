"""
TTB Label Verification — launcher from project root.
Run from project root: streamlit run app.py
Delegates to src.app so all logic stays in src/.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Run the Streamlit app in src so Streamlit’s script runner executes it
import runpy
runpy.run_module("src.app", run_name="__main__")
