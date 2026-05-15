"""
Research logging: scalar metrics in CSV, complex structures in JSON per experiment.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

_SRC = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC.parent
LOGS_DIR = PROJECT_ROOT / "logs"
METRICS_LOG_PATH = LOGS_DIR / "metrics_log.csv"
PARAMS_DIR = LOGS_DIR / "params"

METRICS_FIELDNAMES = [
    "Experiment_ID",
    "Timestamp",
    "Source",
    "Acurácia",
    "F1_Weighted",
    "F1_Estável",
    "Recall_Estável",
    "Alvo_Banda",
    "CV_Janela",
    "CV_max_train_size",
    "N_HMM_STATES",
    "HMM_REFIT",
    "N_Samples",
    "N_Features",
    "Optuna_N_Trials",
    "Optuna_Best_F1_Estável",
]


def _scalar_row(row: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {k: "" for k in METRICS_FIELDNAMES}
    for k in METRICS_FIELDNAMES:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if isinstance(v, (dict, list)):
            raise TypeError(f"metrics_log must be scalar-only; got {type(v)} for {k}")
        if isinstance(v, bool):
            out[k] = str(int(v))
        elif isinstance(v, float):
            out[k] = repr(v) if v != v else ""  # NaN -> empty
        else:
            out[k] = str(v)
    return out


def append_metrics_row(row: Mapping[str, Any]) -> None:
    """Append one row to logs/metrics_log.csv (creates dirs and header if needed)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not METRICS_LOG_PATH.exists()
    line = _scalar_row(row)
    with open(METRICS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(line)


def save_params_json(experiment_id: str, payload: Mapping[str, Any]) -> Path:
    """Write logs/params/{Experiment_ID}_params.json (UTF-8, pretty-printed)."""
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARAMS_DIR / f"{experiment_id}_params.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, default=str)
    return path
