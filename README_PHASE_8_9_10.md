# Phases 8, 9, 10: how to use these files

Unzip over your project root (the `clinical-trial-monitor/` folder). Paths match the project layout.
`app/streamlit_app.py`, `src/ctmon/audit/hashchain.py` and `requirements.txt` replace the placeholder versions.

    pip install -r requirements.txt

## Phase 8: audit trail
    python run_phase8.py          # needs Phase 6 + 7 outputs; builds a demo log, verifies it, attacks copies of it
    pytest tests/test_phase8.py

## Phase 9: dashboard
    python build_dashboard_data.py        # runs the whole pipeline once, writes data/processed/* and the audit log
    streamlit run app/streamlit_app.py
    pytest tests/test_phase9_smoke.py     # headless test on a small fake bundle (no pipeline needed)

Want to look at the dashboard before running the pipeline?

    python -c "import sys; sys.path[:0]=['src','tests']; from fake_bundle import make_fake_bundle; make_fake_bundle('demo_root')"
    set CTMON_ROOT=demo_root   (Windows)   /   export CTMON_ROOT=demo_root   (Mac/Linux)
    streamlit run app/streamlit_app.py

## Phase 10: evaluation
    python run_phase10.py                 # main dataset + 2 untouched replicate datasets (a few minutes)
    python run_phase10.py --seeds 0       # main dataset only
    pytest tests/test_phase10.py tests/test_phase10_report.py

Outputs: docs/evaluation_report.md, docs/LIMITATIONS.md, docs/phase10_main.png, docs/phase10_sensitivity.png,
data/processed/eval/*.csv
