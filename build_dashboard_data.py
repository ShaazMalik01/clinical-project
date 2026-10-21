"""Run the full pipeline once and write everything the dashboard reads (data/processed/*)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ctmon.artifacts import build_artifacts

ap = argparse.ArgumentParser()
ap.add_argument("--reset-audit", action="store_true",
                help="archive the current audit log (never deleted) and start a new chain")
args = ap.parse_args()
m = build_artifacts(ROOT, reset_audit=args.reset_audit)
print("Built dashboard data:")
for k, v in m.items():
    print(f"  {k}: {v}")
print("\nNext: streamlit run app/streamlit_app.py")
