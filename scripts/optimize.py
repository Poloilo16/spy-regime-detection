"""
Bayesian hyperparameter search (Optuna) for the SPY regime / vol-shock pipeline.
Maximizes F1-score of class 1 (Estável) on walk-forward CV (rolling train, 756 days).
Requires: pip install optuna
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import optuna

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ / 'src'))

import training
from experiment_logging import append_metrics_row, save_params_json
from training import (
    DEFAULT_DB_PATH,
    HMM_FEATURES,
    LAGS,
    RV_SHOCK_HIGH,
    RV_SHOCK_LOW,
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

    append_metrics_row({
        'Experiment_ID': log_id,
        'Timestamp': ts,
        'Source': 'optimize',
        'Acurácia': float(rd['accuracy']),
        'F1_Weighted': float(rd['weighted avg']['f1-score']),
        'F1_Estável': float(rd[trn[1]]['f1-score']),
        'Recall_Estável': float(rd[trn[1]]['recall']),
        'Alvo_Banda': float(RV_SHOCK_HIGH),
        'CV_Janela': 'Rolling',
        'CV_max_train_size': int(CV_MAX_TRAIN),
        'N_HMM_STATES': int(bp['N_HMM_STATES']),
        'HMM_REFIT': int(bp['HMM_REFIT']),
        'N_Samples': int(len(yb)),
        'N_Features': int(Xb.shape[1]),
        'Optuna_N_Trials': int(N_TRIALS),
        'Optuna_Best_F1_Estável': float(study.best_value),
    })

    save_params_json(log_id, {
        'run': 'optuna_best_refit',
        'hmm_features': list(HMM_FEATURES),
        'lags': list(LAGS),
        'rv_shock_low': float(RV_SHOCK_LOW),
        'rv_shock_high': float(RV_SHOCK_HIGH),
        'cv_max_train_size': int(CV_MAX_TRAIN),
        'optuna_n_trials': int(N_TRIALS),
        'optuna_best_value_f1_estável': float(study.best_value),
        'optuna_best_params': study.best_params,
        'xgb_classifier_params': xgb_best,
        'classification_report': rd,
    })
    print(f"\nMétricas: {_PROJ / 'logs' / 'metrics_log.csv'}")
    print(f"Parâmetros: {_PROJ / 'logs' / 'params' / (log_id + '_params.json')}")


if __name__ == '__main__':
    main()
