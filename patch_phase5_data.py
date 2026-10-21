"""Adds the CRP lab (Benford-suitable) to data/clean and data/raw, updates ground truth."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from ctmon.data_gen import load_tables, load_yaml, save_tables
from ctmon.data_gen.crp_patch import add_crp

settings = load_yaml(ROOT / "config" / "settings.yaml")
gt_path = ROOT / "data" / "ground_truth" / "anomaly_ground_truth.csv"
sg_path = ROOT / "data" / "ground_truth" / "site_ground_truth.csv"

clean, raw = load_tables(ROOT / "data" / "clean"), load_tables(ROOT / "data" / "raw")
gt, site_gt = pd.read_csv(gt_path), pd.read_csv(sg_path)

clean, raw, gt, site_gt = add_crp(clean, raw, gt, site_gt, seed=settings["seed"] + 5)

save_tables(clean, ROOT / "data" / "clean")
save_tables(raw, ROOT / "data" / "raw")
gt.to_csv(gt_path, index=False)
site_gt.to_csv(sg_path, index=False)

fab = gt[gt["subtype"] == "fabricated_labs"]
print(f"CRP added. Labs columns: {list(raw['labs'].columns)}")
print(f"Ground truth: {len(fab)} 'fabricated_labs' rows at site {fab['site_id'].unique().tolist()}")
