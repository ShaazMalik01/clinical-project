"""One command to (re)build everything:  python run_all.py [--rebuild-data] [--eval]

  1. generate the synthetic trial + injected anomalies + CRP lab   (only if data/raw is missing, or --rebuild-data)
  2. run rules, ML, forensics, site risk, why-flagged cards, and populate the audit log
  3. optionally run the full Phase 10 evaluation (--eval)
Then:  streamlit run app/streamlit_app.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--rebuild-data", action="store_true", help="regenerate the synthetic dataset")
ap.add_argument("--eval", action="store_true", help="also run the Phase 10 evaluation (a few minutes)")
args = ap.parse_args()

if args.rebuild_data or not (ROOT / "data" / "raw" / "subjects.csv").exists():
    from ctmon.data_gen import build_dataset
    print("1/3 Generating the synthetic trial...")
    build_dataset(ROOT / "config" / "settings.yaml", ROOT / "config" / "protocol.yaml")
    subprocess.run([sys.executable, str(ROOT / "patch_phase5_data.py")], check=True)
else:
    print("1/3 Using the existing dataset in data/raw (use --rebuild-data to regenerate).")

print("2/3 Running all detectors and building the dashboard data...")
subprocess.run([sys.executable, str(ROOT / "build_dashboard_data.py"), "--reset-audit"], check=True)

if args.eval:
    print("3/3 Running the final evaluation...")
    subprocess.run([sys.executable, str(ROOT / "run_phase10.py")], check=True)
else:
    print("3/3 Skipped the evaluation (add --eval).")
print("\nDone. Start the dashboard with:  streamlit run app/streamlit_app.py")
