import csv
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import training
from training import (
    DEFAULT_DB_PATH,
    HMM_FEATURES,
    LAGS,
    REGIME_NAMES,
    RV_SHOCK_HIGH,
    TARGET_NAMES,
    build_xy,
    walk_forward_cv_metrics,
)

_ROOT = Path(__file__).resolve().parent
DB_PATH = DEFAULT_DB_PATH

N_HMM_STATES = 4
MIN_HMM = training.MIN_HMM_DEFAULT
HMM_REFIT = 21
XGB_PARAMS = {
    **training.XGB_PARAMS_BASE,
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
}

CV_MAX_TRAIN_SIZE = 756


def main():
    EXP_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

    df, FEATURES, X, y = build_xy(
        DB_PATH, N_HMM_STATES, HMM_REFIT, min_hmm=MIN_HMM, verbose=True,
    )
    print("\nRegime distribution (rv_21d expanding quartiles):")
    print(df['regime'].value_counts().sort_index().rename(REGIME_NAMES))

    print(f"\nTarget (5d RV shock ratio): {training.RV_SHOCK_LOW} / {RV_SHOCK_HIGH}")
    print(df['target'].value_counts().sort_index().rename(TARGET_NAMES))
    print(f"\nFeature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    CV_JANELA = 'Rolling' if CV_MAX_TRAIN_SIZE is not None else 'Expanding'

    cv_out = walk_forward_cv_metrics(
        X, y, XGB_PARAMS, max_train_size=CV_MAX_TRAIN_SIZE,
    )
    all_preds = cv_out['all_preds']
    all_true = cv_out['all_true']
    cv_test_idx = cv_out['cv_test_idx']

    target_report_names = cv_out['target_report_names']
    print("\nWalk-forward CV — Classification Report:")
    print(
        classification_report(
            all_true, all_preds, target_names=target_report_names, zero_division=0,
        )
    )
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(all_true, all_preds))

    final_clf = XGBClassifier(**XGB_PARAMS)
    final_weights = compute_sample_weight(class_weight='balanced', y=y)
    final_clf.fit(X, y, sample_weight=final_weights)

    importance = pd.Series(final_clf.feature_importances_, index=FEATURES)
    print("\nTop 15 feature importances:")
    print(importance.nlargest(15).round(4).to_string())

    x_latest = df[FEATURES].iloc[[-1]].values
    pred_proba = final_clf.predict_proba(x_latest)[0]
    pred_class = int(np.argmax(pred_proba))
    latest_date = df['date'].iloc[-1]
    today_regime = int(df['regime_lag1'].iloc[-1])

    print(f"\n{'─'*45}")
    print(f"Latest date  : {latest_date.date()}")
    print(f"Today regime : {today_regime} ({REGIME_NAMES[today_regime]})")
    print(f"Predicted    : {TARGET_NAMES[pred_class]} (class {pred_class})")
    print(f"Probabilities: " + "  ".join(f"{TARGET_NAMES[i]}={p:.2f}" for i, p in enumerate(pred_proba)))
    print(f"{'─'*45}")

    explainer_shap = shap.TreeExplainer(final_clf)
    shap_raw = explainer_shap.shap_values(x_latest)
    if isinstance(shap_raw, list):
        shap_last_class = np.asarray(shap_raw[pred_class], dtype=float).reshape(-1)
    else:
        shap_arr = np.asarray(shap_raw, dtype=float)
        if shap_arr.ndim == 3:
            shap_last_class = shap_arr[0, :, pred_class]
        elif shap_arr.ndim == 2:
            shap_last_class = shap_arr[0]
        else:
            shap_last_class = shap_arr.ravel()

    abs_order = np.argsort(np.abs(shap_last_class))[::-1][:5]
    print(
        f"\nSHAP (TreeExplainer) — top 5 impactos na classe prevista "
        f"({TARGET_NAMES[pred_class]}), última data:"
    )
    for rank, j in enumerate(abs_order, start=1):
        name = FEATURES[int(j)]
        print(f"  {rank}. {name}: {shap_last_class[j]:+.6f}")

    logs_dir = _ROOT / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    experiment_log_path = logs_dir / 'experiment_log.csv'

    report_dict = classification_report(
        all_true, all_preds,
        target_names=target_report_names,
        zero_division=0,
        output_dict=True,
    )
    accuracy = float(report_dict['accuracy'])
    recall_estavel = float(report_dict[target_report_names[1]]['recall'])
    f1_weighted = float(report_dict['weighted avg']['f1-score'])
    top5_features = ','.join(importance.nlargest(5).index.astype(str).tolist())
    cv_max_train_str = '' if CV_MAX_TRAIN_SIZE is None else str(int(CV_MAX_TRAIN_SIZE))

    log_row = [
        EXP_ID,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ','.join(HMM_FEATURES),
        str(LAGS),
        float(RV_SHOCK_HIGH),
        CV_JANELA,
        cv_max_train_str,
        accuracy,
        recall_estavel,
        f1_weighted,
        top5_features,
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

    preds_wf_path = logs_dir / f'preds_{EXP_ID}.csv'
    preds_wf = pd.DataFrame(
        {
            'date': df.iloc[cv_test_idx]['date'].to_numpy(),
            'log_return': df.iloc[cv_test_idx]['log_return'].to_numpy(),
            'actual_target': all_true,
            'predicted_target': all_preds,
        }
    )
    preds_wf.to_csv(preds_wf_path, index=False, encoding='utf-8')

    write_header = not experiment_log_path.exists()
    with open(experiment_log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(log_row)


if __name__ == '__main__':
    main()
