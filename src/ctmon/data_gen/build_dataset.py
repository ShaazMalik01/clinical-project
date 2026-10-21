"""One-call dataset builder: clean data + injected data + ground truth."""
from __future__ import annotations

from pathlib import Path

from .anomaly_injector import AnomalyInjector
from .generator import generate_clean_trial, load_yaml, save_tables


def build_dataset(settings_path: str | Path, protocol_path: str | Path):
    settings_path = Path(settings_path)
    root = settings_path.resolve().parents[1]
    settings, protocol = load_yaml(settings_path), load_yaml(protocol_path)

    clean = generate_clean_trial(settings, protocol)
    injector = AnomalyInjector(clean, protocol, settings, seed=settings["seed"] + 1)
    dirty, gt, site_gt = injector.run()

    save_tables(clean, root / "data" / "clean")
    save_tables(dirty, root / "data" / "raw")
    (root / "data" / "ground_truth").mkdir(parents=True, exist_ok=True)
    gt.to_csv(root / "data" / "ground_truth" / "anomaly_ground_truth.csv", index=False)
    site_gt.to_csv(root / "data" / "ground_truth" / "site_ground_truth.csv", index=False)
    return clean, dirty, gt, site_gt
