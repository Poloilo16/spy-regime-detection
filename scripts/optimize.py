"""
Bayesian hyperparameter search (Optuna) for the SPY regime / vol-shock pipeline.
Maximizes F1-score of class 1 (Estável) on walk-forward CV (rolling train, 756 days).
Requires: pip install optuna
"""
from __future__ import annotations

import csv
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import optuna
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from training import (
    DEFAULT_DB_PATH,
    HMM_FEATURES,
    LAGS,
    RV_SHOCK_HIGH,
    build_xy,
    walk_forward_cv_metrics,
)
N_TRIALS = 50
CV_MAX_TRAIN = 756
XGB_PARAMS_BASE = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0,
)

def _xgb_params_from_trial(trial: optuna.Trial) -> dict:
    return {
        **XGB_PARAMS_BASE,
        'max_depth': trial.suggest_categorical('XGB_MAX_DEPTH', [3, 4, 5, 6]),
        'learning_rate': trial.suggest_categorical('XGB_LEARNING_RATE', [0.01, 0.05, 0.1]),
        'n_estimators': trial.suggest_categorical('XGB_N_ESTIMATORS', [100, 300, 500]),
    }


def objective(trial: optuna.Trial) -> float:
    n_hmm = trial.suggest_categorical('N_HMM_STATES', [2, 3, 4])
    hmm_refit = trial.suggest_categorical('HMM_REFIT', [10, 21, 42, 63])
    xgb_params = _xgb_params_from_trial(trial)
    try:
        _, _, X, y = build_xy(DEFAULT_DB_PATH, n_hmm, hmm_refit, verbose=False)
        if len(y) < 400:
            return float('-inf')
        m = walk_forward_cv_metrics(X, y, xgb_params, max_train_size=CV_MAX_TRAIN)
    except Exception:
        return float('-inf')
    trial.set_user_attr('f1_weighted', m['f1_weighted'])
    trial.set_user_attr('accuracy', m['accuracy'])
    return m['f1_stable']


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print('\n' + '=' * 60)
    print('Best trial:')
    print(f"  value (F1 Estável, classe 1): {study.best_value:.4f}")
    print('  params:', json.dumps(study.best_params, indent=2, sort_keys=True))
    if study.best_trial.user_attrs:
        print('  user_attrs:', study.best_trial.user_attrs)
    print('=' * 60)

    bp = study.best_params
    xgb_best = {
        **XGB_PARAMS_BASE,
        'max_depth': bp['XGB_MAX_DEPTH'],
        'learning_rate': bp['XGB_LEARNING_RATE'],
        'n_estimators': bp['XGB_N_ESTIMATORS'],
    }
    _, _, Xb, yb = build_xy(DEFAULT_DB_PATH, bp['N_HMM_STATES'], bp['HMM_REFIT'], verbose=False)
    m_best = walk_forward_cv_metrics(Xb, yb, xgb_best, max_train_size=CV_MAX_TRAIN)
    rd = m_best['report_dict']
    trn = m_best['target_report_names']

    log_id = f"OPTUNA_BEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hmm_meta = f"{','.join(HMM_FEATURES)}|n_states={bp['N_HMM_STATES']}|HMM_REFIT={bp['HMM_REFIT']}"

    log_row = [
        log_id,
        ts,
        hmm_meta,
        str(LAGS),
        float(RV_SHOCK_HIGH),
        'Rolling',
        str(int(CV_MAX_TRAIN)),
        float(rd['accuracy']),
        float(rd[trn[1]]['recall']),
        float(rd['weighted avg']['f1-score']),
        json.dumps(study.best_params, sort_keys=True),
    ]
    header = [
        'Experiment_ID',
        'Timestamp',
        'HMM_FEATURES',
        'Lags',
        'Alvo_Banda',
        'CV_Janela',
        'CV_max_train_size',
        'Acurácia',
        'Recall_Estável',
        'F1_Weighted',
        'Top_5_Features',
    ]

    logs_dir = _REPO_ROOT / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    experiment_log_path = logs_dir / 'experiment_log.csv'
    write_header = not experiment_log_path.exists()
    with open(experiment_log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(log_row)
    print(f"\nLinha Optuna gravada em: {experiment_log_path}")


if __name__ == '__main__':
    main()
