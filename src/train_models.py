# """Train leakage-aware congestion and delay models on the synthetic Aiqon dataset."""
# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
#                              classification_report, confusion_matrix, roc_auc_score,
#                              precision_score, recall_score)
# import joblib

# ROOT = Path(__file__).resolve().parents[1]
# DATA = ROOT / 'data' / 'realistic_port_congestion_800.csv'
# MODEL_DIR = ROOT / 'models'
# REPORT_DIR = ROOT / 'reports'
# MODEL_DIR.mkdir(exist_ok=True)
# REPORT_DIR.mkdir(exist_ok=True)

# # These are outcomes or post-prediction observations and must never be features.
# LEAKAGE_COLUMNS = [
#     'observed_queue_length_vessels',
#     'observed_waiting_time_hours',
#     'observed_shipment_delay_hours',
#     'congestion_level',
#     'shipment_delayed_6h_or_more',
#     'date',
# ]
# TARGETS = ['congestion_level', 'shipment_delayed_6h_or_more']


# def make_preprocessor(X):
#     cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
#     num_cols = [c for c in X.columns if c not in cat_cols]
#     numeric = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler()),
#     ])
#     categorical = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore')),
#     ])
#     return ColumnTransformer([
#         ('numeric', numeric, num_cols),
#         ('categorical', categorical, cat_cols),
#     ])


# def chronological_masks(df):
#     dates = np.sort(df['date'].dropna().unique())
#     cut_train = int(len(dates) * 0.70)
#     cut_val = int(len(dates) * 0.85)
#     train_dates = set(dates[:cut_train])
#     val_dates = set(dates[cut_train:cut_val])
#     test_dates = set(dates[cut_val:])
#     return df['date'].isin(train_dates), df['date'].isin(val_dates), df['date'].isin(test_dates), dates, cut_train, cut_val


# def evaluate(y_true, y_pred, labels):
#     return {
#         'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
#         'balanced_accuracy': round(float(balanced_accuracy_score(y_true, y_pred)), 4),
#         'macro_f1': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
#         'weighted_f1': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
#         'labels_in_confusion_matrix': [str(x) for x in labels],
#         'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
#         'classification_report': classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
#     }


# def main():
#     df = pd.read_csv(DATA)
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     if df['date'].isna().any():
#         raise ValueError('Some dates could not be parsed. Fix date values before training.')
#     if df.duplicated().any():
#         print(f'WARNING: {int(df.duplicated().sum())} exact duplicate rows found.')

#     train_mask, val_mask, test_mask, dates, cut_train, cut_val = chronological_masks(df)
#     X = df.drop(columns=LEAKAGE_COLUMNS)
#     y_cong = df['congestion_level'].astype(str)
#     y_delay = df['shipment_delayed_6h_or_more'].astype(int)

#     # Models are trained on the earlier 70%; validation period is reserved for later tuning.
#     # The final 15% is a held-out chronological test period.
#     model_specs = {
#         'logistic_regression': LogisticRegression(max_iter=3000, class_weight='balanced'),
#         'random_forest': RandomForestClassifier(
#             n_estimators=500, min_samples_leaf=2, max_features='sqrt',
#             class_weight='balanced_subsample', random_state=42, n_jobs=-1,
#         ),
#     }
#     all_metrics = {
#         'dataset': {
#             'synthetic': True,
#             'rows': int(len(df)),
#             'columns': int(df.shape[1]),
#             'date_start': str(df['date'].min().date()),
#             'date_end': str(df['date'].max().date()),
#             'missing_values_by_column': {k: int(v) for k, v in df.isna().sum().items() if v > 0},
#             'exact_duplicate_rows': int(df.duplicated().sum()),
#             'congestion_class_counts': {str(k): int(v) for k, v in y_cong.value_counts().items()},
#             'delay_class_counts': {str(k): int(v) for k, v in y_delay.value_counts().items()},
#         },
#         'split': {
#             'method': 'chronological by unique date; no random shuffling',
#             'train_dates': [str(dates[0])[:10], str(dates[cut_train-1])[:10]],
#             'validation_dates': [str(dates[cut_train])[:10], str(dates[cut_val-1])[:10]],
#             'test_dates': [str(dates[cut_val])[:10], str(dates[-1])[:10]],
#             'train_rows': int(train_mask.sum()),
#             'validation_rows': int(val_mask.sum()),
#             'test_rows': int(test_mask.sum()),
#         },
#         'feature_columns': list(X.columns),
#         'targets': {},
#     }

#     for target_name, y in [('congestion_level', y_cong), ('shipment_delayed_6h_or_more', y_delay)]:
#         all_metrics['targets'][target_name] = {}
#         for model_name, estimator in model_specs.items():
#             pipeline = Pipeline([('preprocessor', make_preprocessor(X)), ('model', estimator)])
#             pipeline.fit(X.loc[train_mask], y.loc[train_mask])
#             pred_val = pipeline.predict(X.loc[val_mask])
#             pred_test = pipeline.predict(X.loc[test_mask])
#             classes = list(pipeline.named_steps['model'].classes_)
#             metrics = {
#                 'validation': evaluate(y.loc[val_mask], pred_val, classes),
#                 'test': evaluate(y.loc[test_mask], pred_test, classes),
#             }
#             if target_name == 'shipment_delayed_6h_or_more' and hasattr(pipeline, 'predict_proba') and 1 in classes:
#                 proba = pipeline.predict_proba(X.loc[test_mask])[:, classes.index(1)]
#                 metrics['test']['roc_auc'] = round(float(roc_auc_score(y.loc[test_mask], proba)), 4) if y.loc[test_mask].nunique() == 2 else None
#                 metrics['test']['delay_precision'] = round(float(precision_score(y.loc[test_mask], pred_test, pos_label=1, zero_division=0)), 4)
#                 metrics['test']['delay_recall'] = round(float(recall_score(y.loc[test_mask], pred_test, pos_label=1, zero_division=0)), 4)
#             all_metrics['targets'][target_name][model_name] = metrics
#             joblib.dump(pipeline, MODEL_DIR / f'{target_name}__{model_name}.joblib')

#             # Save row-level held-out predictions for review.
#             out = df.loc[test_mask, ['port_name', 'date']].copy()
#             out['actual'] = y.loc[test_mask].values
#             out['predicted'] = pred_test
#             if hasattr(pipeline, 'predict_proba'):
#                 probabilities = pipeline.predict_proba(X.loc[test_mask])
#                 out['max_class_probability'] = probabilities.max(axis=1)
#                 if target_name == 'shipment_delayed_6h_or_more' and 1 in classes:
#                     out['predicted_delay_probability'] = probabilities[:, classes.index(1)]
#             out.to_csv(REPORT_DIR / f'{target_name}__{model_name}__test_predictions.csv', index=False)

#     # A transparent data-quality summary, not a model result.
#     data_summary = {
#         'shape': [int(df.shape[0]), int(df.shape[1])],
#         'missing_values': {k: int(v) for k, v in df.isna().sum().items() if v > 0},
#         'duplicate_rows': int(df.duplicated().sum()),
#         'congestion_counts': {str(k): int(v) for k, v in y_cong.value_counts().items()},
#         'delay_counts': {str(k): int(v) for k, v in y_delay.value_counts().items()},
#         'congestion_summary': df.groupby('congestion_level')[['observed_queue_length_vessels','observed_waiting_time_hours','observed_shipment_delay_hours']].mean().round(3).to_dict(),
#     }
#     with open(REPORT_DIR / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
#         json.dump(all_metrics, f, indent=2, default=str)
#     with open(REPORT_DIR / 'data_quality_summary.json', 'w', encoding='utf-8') as f:
#         json.dump(data_summary, f, indent=2, default=str)

#     print('Chronological split:', all_metrics['split'])
#     print('Synthetic data quality:', json.dumps(data_summary, indent=2))
#     for target, models in all_metrics['targets'].items():
#         print('\nTARGET:', target)
#         for model, result in models.items():
#             print(model, 'TEST:', {k: result['test'].get(k) for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1','roc_auc','delay_precision','delay_recall'] if k in result['test']})
#     print('\nSaved models to', MODEL_DIR)
#     print('Saved reports to', REPORT_DIR)

# if __name__ == '__main__':
#     main()


# """Leakage-aware, validation-driven training for the synthetic port-congestion dataset (v2).



# What changed versus train_models.py

#   1. The validation period is actually used: feature set, model family and

#      regularisation are chosen on validation only (threshold-free ROC-AUC).

#   2. The decision threshold (High congestion / delay) is tuned on validation only.

#   3. The test period is scored once, for the selected model, next to the original

#      baseline (Logistic Regression, all features, default C).

#   4. Per-port metrics and bootstrap 95% confidence intervals are reported, because

#      the test period has very few High cases.

#   5. Rolling-origin (expanding window) cross-validation gives a steadier estimate

#      than a single split. Hyperparameters are fixed there, and no tuned threshold

#      is applied, so it is a comparison of model families, not a nested evaluation.



# The data is synthetic. Results describe this dataset, not real port operations.



# Usage:

#     python train_models_v2.py                       # data in ../data/, outputs in ../

#     python train_models_v2.py --data path/to.csv --out some/dir

#     python train_models_v2.py --refit               # also report a train+val refit

#     python train_models_v2.py --beta 2              # tune thresholds to favour recall

# """

# from __future__ import annotations



# import argparse

# import json

# import warnings

# from pathlib import Path



# import joblib

# import numpy as np

# import pandas as pd

# from sklearn.compose import ColumnTransformer

# from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import (accuracy_score, balanced_accuracy_score,

#                              confusion_matrix, f1_score, fbeta_score,

#                              precision_score, recall_score, roc_auc_score)

# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import OneHotEncoder, StandardScaler



# warnings.filterwarnings("ignore")



# # Outcomes / post-prediction observations: never features.

# LEAKAGE_COLUMNS = [

#     "observed_queue_length_vessels",

#     "observed_waiting_time_hours",

#     "observed_shipment_delay_hours",

#     "congestion_level",

#     "shipment_delayed_6h_or_more",

#     "date",

# ]

# CALENDAR = ["month", "day_of_week", "is_weekend", "is_public_holiday"]

# VESSEL_MIX = ["scheduled_container_vessels", "scheduled_bulk_vessels",

#               "scheduled_tanker_vessels", "scheduled_other_vessels"]

# # Candidate feature sets, expressed as columns to DROP (validation decides).

# FEATURE_SETS = {

#     "all": [],

#     "no_calendar": CALENDAR,

#     "compact": CALENDAR + VESSEL_MIX,

# }

# LR_C_GRID = (0.05, 0.2, 1.0)

# THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)

# N_BOOT = 2000

# SEED = 42





# # --------------------------------------------------------------------------- utils

# def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

#     cat_cols = X.select_dtypes(exclude="number").columns.tolist()

#     num_cols = [c for c in X.columns if c not in cat_cols]

#     numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),

#                         ("scaler", StandardScaler())])

#     categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),

#                             ("onehot", OneHotEncoder(handle_unknown="ignore"))])

#     return ColumnTransformer([("numeric", numeric, num_cols),

#                               ("categorical", categorical, cat_cols)])





# def chronological_masks(df: pd.DataFrame):

#     dates = np.sort(df["date"].unique())

#     c1, c2 = int(len(dates) * 0.70), int(len(dates) * 0.85)

#     return (df["date"].isin(dates[:c1]), df["date"].isin(dates[c1:c2]),

#             df["date"].isin(dates[c2:]), dates, c1, c2)





# def make_candidates():

#     """(name, feature_set, estimator_factory) for every model/feature-set pair."""

#     cands = []

#     for fs in FEATURE_SETS:

#         for C in LR_C_GRID:

#             cands.append((f"LR|{fs}|C={C}", fs, lambda C=C: LogisticRegression(

#                 max_iter=3000, class_weight="balanced", C=C)))

#         cands.append((f"HGB|{fs}", fs, lambda: HistGradientBoostingClassifier(

#             class_weight="balanced", max_depth=3, learning_rate=0.05,

#             max_iter=120, random_state=SEED)))

#         cands.append((f"RF|{fs}", fs, lambda: RandomForestClassifier(

#             n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",

#             random_state=SEED, n_jobs=-1)))

#     return cands





# def features_for(df: pd.DataFrame, fs: str) -> pd.DataFrame:

#     return df.drop(columns=LEAKAGE_COLUMNS + FEATURE_SETS[fs])





# def fit_pipeline(df, mask, fs, factory, y):

#     X = features_for(df, fs)

#     pipe = Pipeline([("preprocessor", make_preprocessor(X)), ("model", factory())])

#     pipe.fit(X.loc[mask], y.loc[mask])

#     return pipe, X





# # ------------------------------------------------------------------ decisions

# def decide_congestion(proba, classes, t_high):

#     """Predict High when P(High) >= t_high, otherwise the likelier of Low/Moderate."""

#     hi = classes.index("High")

#     others = [i for i in range(len(classes)) if i != hi]

#     best_other = np.array([classes[others[int(np.argmax(p[others]))]] for p in proba],

#                           dtype=object)

#     return np.where(proba[:, hi] >= t_high, "High", best_other)





# def tune_threshold(y_true, score_fn):

#     """Pick the threshold with the best F-beta on validation (ties -> closest to 0.5)."""

#     best = max(THRESHOLD_GRID, key=lambda t: (score_fn(t), -abs(t - 0.5)))

#     return float(best)





# # -------------------------------------------------------------------- metrics

# def _safe_div(a, b):

#     return float(a) / b if b else float("nan")





# def cong_metrics(y, pred, classes, ports=None):

#     y, pred = np.asarray(y, dtype=object), np.asarray(pred, dtype=object)

#     out = {

#         "accuracy": round(float(accuracy_score(y, pred)), 4),

#         "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),

#         "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),

#         "high_recall": round(float(recall_score(y == "High", pred == "High", zero_division=0)), 4),

#         "high_precision": round(float(precision_score(y == "High", pred == "High", zero_division=0)), 4),

#         "n_high": int((y == "High").sum()),

#         "labels": classes,

#         "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),

#     }

#     if ports is not None:

#         ports = np.asarray(ports)

#         out["per_port"] = {}

#         for p in sorted(set(ports)):

#             m = ports == p

#             out["per_port"][p] = {

#                 "n": int(m.sum()), "n_high": int((y[m] == "High").sum()),

#                 "high_recall": round(float(recall_score(y[m] == "High", pred[m] == "High", zero_division=0)), 3),

#                 "high_precision": round(float(precision_score(y[m] == "High", pred[m] == "High", zero_division=0)), 3),

#                 "accuracy": round(float(accuracy_score(y[m], pred[m])), 3),

#             }

#     return out





# def delay_metrics(y, pred, proba=None, ports=None):

#     y, pred = np.asarray(y), np.asarray(pred)

#     out = {

#         "accuracy": round(float(accuracy_score(y, pred)), 4),

#         "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),

#         "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),

#         "delay_precision": round(float(precision_score(y, pred, zero_division=0)), 4),

#         "delay_recall": round(float(recall_score(y, pred, zero_division=0)), 4),

#         "n_delayed": int(y.sum()),

#         "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),

#     }

#     if proba is not None and len(set(y)) == 2:

#         out["roc_auc"] = round(float(roc_auc_score(y, proba)), 4)

#     if ports is not None:

#         ports = np.asarray(ports)

#         out["per_port"] = {}

#         for p in sorted(set(ports)):

#             m = ports == p

#             out["per_port"][p] = {

#                 "n": int(m.sum()), "n_delayed": int(y[m].sum()),

#                 "delay_recall": round(float(recall_score(y[m], pred[m], zero_division=0)), 3),

#                 "delay_precision": round(float(precision_score(y[m], pred[m], zero_division=0)), 3),

#             }

#     return out





# def bootstrap_ci(y, pred, fn, n=N_BOOT, seed=0):

#     """95% percentile bootstrap CI over test rows (resamples with no positives are skipped)."""

#     rng = np.random.default_rng(seed)

#     y, pred = np.asarray(y, dtype=object), np.asarray(pred, dtype=object)

#     vals = []

#     for _ in range(n):

#         i = rng.integers(0, len(y), len(y))

#         v = fn(y[i], pred[i])

#         if not np.isnan(v):

#             vals.append(v)

#     lo, hi = np.percentile(vals, [2.5, 97.5])

#     return [round(float(lo), 3), round(float(hi), 3)]





# def ci_block_cong(y, pred):

#     return {

#         "balanced_accuracy_95ci": bootstrap_ci(y, pred, lambda a, b: balanced_accuracy_score(a, b)),

#         "high_recall_95ci": bootstrap_ci(

#             y, pred, lambda a, b: _safe_div(((a == "High") & (b == "High")).sum(), (a == "High").sum())),

#     }





# def ci_block_delay(y, pred):

#     y, pred = np.asarray(y, dtype=int).astype(object), np.asarray(pred, dtype=int).astype(object)

#     return {

#         "balanced_accuracy_95ci": bootstrap_ci(y, pred, lambda a, b: balanced_accuracy_score(a.astype(int), b.astype(int))),

#         "delay_recall_95ci": bootstrap_ci(

#             y, pred, lambda a, b: _safe_div(((a == 1) & (b == 1)).sum(), (a == 1).sum())),

#         "delay_precision_95ci": bootstrap_ci(

#             y, pred, lambda a, b: _safe_div(((a == 1) & (b == 1)).sum(), (b == 1).sum())),

#     }





# # ----------------------------------------------------------------- selection

# def val_score(task, y_val, proba, classes):

#     if task == "congestion":

#         return roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)

#     return roc_auc_score(y_val, proba[:, classes.index(1)])





# def select_model(df, y, task, masks):

#     train_mask, val_mask, _ = masks

#     rows = []

#     for name, fs, factory in make_candidates():

#         pipe, X = fit_pipeline(df, train_mask, fs, factory, y)

#         classes = list(pipe.named_steps["model"].classes_)

#         proba = pipe.predict_proba(X.loc[val_mask])

#         rows.append({"name": name, "feature_set": fs, "factory": factory,

#                      "val_auc": float(val_score(task, y.loc[val_mask], proba, classes))})

#     rows.sort(key=lambda r: r["val_auc"], reverse=True)

#     return rows





# # ------------------------------------------------------------- rolling CV

# def rolling_cv(df, y, task, configs, n_folds=5, start=0.45):

#     dates = np.sort(df["date"].unique())

#     n = len(dates)

#     edges = np.linspace(int(n * start), n, n_folds + 1).astype(int)

#     results = {}

#     for name, fs, factory in configs:

#         pred = pd.Series(index=df.index, dtype=object)

#         proba1 = pd.Series(index=df.index, dtype=float)

#         for i in range(n_folds):

#             tr = df["date"].isin(dates[:edges[i]])

#             te = df["date"].isin(dates[edges[i]:edges[i + 1]])

#             pipe, X = fit_pipeline(df, tr, fs, factory, y)

#             pred.loc[te] = pipe.predict(X.loc[te])

#             if task == "delay":

#                 classes = list(pipe.named_steps["model"].classes_)

#                 proba1.loc[te] = pipe.predict_proba(X.loc[te])[:, classes.index(1)]

#         k = pred.notna()

#         if task == "congestion":

#             results[name] = cong_metrics(y[k], pred[k], ["High", "Low", "Moderate"],

#                                          ports=df.loc[k, "port_name"])

#         else:

#             results[name] = delay_metrics(y[k].astype(int), pred[k].astype(int),

#                                           proba1[k], ports=df.loc[k, "port_name"])

#         results[name]["n_rows"] = int(k.sum())

#     return results





# # ------------------------------------------------------------------- main

# def run_task(task, df, y, masks, args, out_dir):

#     train_mask, val_mask, test_mask = masks

#     ports_test = df.loc[test_mask, "port_name"].values

#     ranking = select_model(df, y, task, masks)

#     best = ranking[0]

#     print(f"\n=== {task}: top validation candidates (ROC-AUC) ===")

#     for r in ranking[:5]:

#         print(f"  {r['val_auc']:.4f}  {r['name']}")



#     # Fit selected config on TRAIN only; tune threshold on VALIDATION only.

#     pipe, X = fit_pipeline(df, train_mask, best["feature_set"], best["factory"], y)

#     classes = list(pipe.named_steps["model"].classes_)

#     p_val, p_test = pipe.predict_proba(X.loc[val_mask]), pipe.predict_proba(X.loc[test_mask])



#     if task == "congestion":

#         yv = y.loc[val_mask].values.astype(object)

#         # Keep congestion threshold tuning unchanged (F-beta controlled by --beta).
#         thr = tune_threshold(yv, lambda t: fbeta_score(
#             yv == "High",
#             decide_congestion(p_val, classes, t) == "High",
#             beta=args.beta,
#             zero_division=0
#         ))

#         pred_val = decide_congestion(p_val, classes, thr)

#         pred_test = decide_congestion(p_test, classes, thr)

#         yt = y.loc[test_mask].values.astype(object)

#         val_m = cong_metrics(yv, pred_val, classes)

#         test_m = cong_metrics(yt, pred_test, classes, ports=ports_test)

#         test_m.update(ci_block_cong(yt, pred_test))

#         test_m["decision_rule"] = f"High if P(High) >= {thr}, else likelier of Low/Moderate"

#     else:

#         yv = y.loc[val_mask].values.astype(int)

#         pv, pt = p_val[:, classes.index(1)], p_test[:, classes.index(1)]

#         # Tune delay threshold separately to favour recall when --delay-beta > 1.
#         thr = tune_threshold(yv, lambda t: fbeta_score(
#             yv,
#             (pv >= t).astype(int),
#             beta=args.delay_beta,
#             zero_division=0
#         ))

#         pred_val, pred_test = (pv >= thr).astype(int), (pt >= thr).astype(int)

#         yt = y.loc[test_mask].values.astype(int)

#         val_m = delay_metrics(yv, pred_val, pv)

#         test_m = delay_metrics(yt, pred_test, pt, ports=ports_test)

#         test_m.update(ci_block_delay(yt, pred_test))

#         test_m["decision_rule"] = f"Delayed if P(delay) >= {thr}"

#         # Also show the default 0.5 operating point for transparency.

#         test_m["at_threshold_0.5"] = delay_metrics(yt, (pt >= 0.5).astype(int), pt)



#     # Original baseline for a like-for-like comparison on the same test rows.

#     base_pipe, Xb = fit_pipeline(df, train_mask, "all", lambda: LogisticRegression(

#         max_iter=3000, class_weight="balanced"), y)

#     pb = base_pipe.predict(Xb.loc[test_mask])

#     if task == "congestion":

#         baseline = cong_metrics(yt, pb.astype(object), classes, ports=ports_test)

#     else:

#         baseline = delay_metrics(yt, pb.astype(int),

#                                  base_pipe.predict_proba(Xb.loc[test_mask])[:, 1], ports=ports_test)



#     result = {

#         "selected": best["name"], "threshold": thr,

#         "validation_ranking": [{"name": r["name"], "val_auc": round(r["val_auc"], 4)} for r in ranking],

#         "validation_metrics": val_m, "test_metrics": test_m,

#         "baseline_original_lr_test": baseline,

#     }



#     if args.refit:

#         trval = train_mask | val_mask

#         pipe2, X2 = fit_pipeline(df, trval, best["feature_set"], best["factory"], y)

#         p2 = pipe2.predict_proba(X2.loc[test_mask])

#         if task == "congestion":

#             pr2 = decide_congestion(p2, classes, thr)

#             result["test_metrics_refit_train_plus_val"] = cong_metrics(yt, pr2, classes, ports=ports_test)

#         else:

#             q2 = p2[:, classes.index(1)]

#             result["test_metrics_refit_train_plus_val"] = delay_metrics(yt, (q2 >= thr).astype(int), q2, ports=ports_test)

#         result["refit_note"] = ("Hyperparameters and threshold come from the train-only fit; "

#                                 "the refit model's probabilities may be calibrated slightly differently.")



#     # Rolling CV: baseline vs selected vs other families (fixed hyperparameters).

#     cv_cfgs = [("baseline_LR|all|C=1.0", "all", lambda: LogisticRegression(

#                    max_iter=3000, class_weight="balanced")),

#                (best["name"] + " (selected)", best["feature_set"], best["factory"])]

#     for r in ranking:

#         fam = r["name"].split("|")[0]

#         if fam in ("HGB", "RF") and not any(c[0].startswith(fam) for c in cv_cfgs):

#             cv_cfgs.append((r["name"], r["feature_set"], r["factory"]))

#     result["rolling_cv"] = rolling_cv(df, y, task, cv_cfgs)



#     # Save artefacts.

#     joblib.dump({"pipeline": pipe, "classes": classes, "threshold": thr,

#                  "config": best["name"], "feature_columns": list(X.columns)},

#                 out_dir / "models" / f"{task}__v2_selected.joblib")

#     pred_df = df.loc[test_mask, ["port_name", "date"]].copy()

#     pred_df["actual"] = yt

#     pred_df["predicted"] = pred_test

#     for j, c in enumerate(classes):

#         pred_df[f"proba_{c}"] = p_test[:, j]

#     pred_df.to_csv(out_dir / "reports" / f"{task}__v2_test_predictions.csv", index=False)

#     return result





# def print_summary(task, res):

#     t, b = res["test_metrics"], res["baseline_original_lr_test"]

#     print(f"\n##### {task.upper()} -- selected: {res['selected']}  (threshold {res['threshold']})")

#     if task == "congestion":

#         keys = ["accuracy", "balanced_accuracy", "macro_f1", "high_recall", "high_precision"]

#     else:

#         keys = ["accuracy", "balanced_accuracy", "macro_f1", "delay_recall", "delay_precision", "roc_auc"]

#     print(f"{'metric':20s} {'baseline LR':>12s} {'selected':>10s}")

#     for k in keys:

#         print(f"{k:20s} {b.get(k, float('nan')):12.3f} {t.get(k, float('nan')):10.3f}")

#     ci_keys = [k for k in t if k.endswith("_95ci")]

#     for k in ci_keys:

#         print(f"  {k}: {t[k]}")

#     print("  per-port (test):")

#     for p, v in t["per_port"].items():

#         print("   ", p, v)

#     if "test_metrics_refit_train_plus_val" in res:

#         r = res["test_metrics_refit_train_plus_val"]

#         print("  refit on train+val, same threshold (test):",

#               {k: r[k] for k in keys if k in r})

#     print("  rolling-origin CV (pooled over future folds):")

#     for name, m in res["rolling_cv"].items():

#         if task == "congestion":

#             print(f"    {name:34s} bal={m['balanced_accuracy']:.3f} mF1={m['macro_f1']:.3f} "

#                   f"HighRec={m['high_recall']:.2f} HighPrec={m['high_precision']:.2f} "

#                   f"(n_high={m['n_high']})")

#             print("       High recall per port:",

#                   {p: v["high_recall"] for p, v in m["per_port"].items()})

#         else:

#             print(f"    {name:34s} bal={m['balanced_accuracy']:.3f} AUC={m.get('roc_auc', float('nan')):.3f} "

#                   f"DelayRec={m['delay_recall']:.2f} DelayPrec={m['delay_precision']:.2f}")





# def main():

#     root = Path(__file__).resolve().parents[1]

#     ap = argparse.ArgumentParser()

#     ap.add_argument("--data", default=str(root / "data" / "realistic_port_congestion_800.csv"))

#     ap.add_argument("--out", default=str(root))

#     ap.add_argument("--refit", action="store_true", help="also report a train+val refit on test")

#     ap.add_argument("--beta", type=float, default=1.0,

#                     help="F-beta used to tune thresholds on validation: 1 balances precision and "

#                          "recall, 2 favours recall (fewer missed High/delayed cases, more false alarms)")

#     ap.add_argument(

#         "--delay-beta",

#         type=float,

#         default=2.0,

#         help="F-beta for shipment-delay threshold tuning; higher values favour recall"

#     )

#     args = ap.parse_args()



#     out_dir = Path(args.out)

#     (out_dir / "models").mkdir(parents=True, exist_ok=True)

#     (out_dir / "reports").mkdir(parents=True, exist_ok=True)



#     df = pd.read_csv(args.data)

#     df["date"] = pd.to_datetime(df["date"], errors="coerce")

#     if df["date"].isna().any():

#         raise ValueError("Some dates could not be parsed.")

#     df = df.sort_values(["date", "port_name"]).reset_index(drop=True)

#     train_mask, val_mask, test_mask, dates, c1, c2 = chronological_masks(df)

#     masks = (train_mask, val_mask, test_mask)



#     y_cong = df["congestion_level"].astype(str)

#     y_delay = df["shipment_delayed_6h_or_more"].astype(int)



#     split = {

#         "train": [str(dates[0])[:10], str(dates[c1 - 1])[:10], int(train_mask.sum())],

#         "validation": [str(dates[c1])[:10], str(dates[c2 - 1])[:10], int(val_mask.sum())],

#         "test": [str(dates[c2])[:10], str(dates[-1])[:10], int(test_mask.sum())],

#         "High_counts": {k: int((y_cong[m] == "High").sum()) for k, m in

#                         [("train", train_mask), ("validation", val_mask), ("test", test_mask)]},

#     }

#     print("Split [start, end, rows]:", json.dumps(split))



#     metrics = {"dataset": {"synthetic": True, "rows": int(len(df))}, "split": split,

#                "feature_sets_dropped_columns": FEATURE_SETS,

#                "targets": {

#                    "congestion_level": run_task("congestion", df, y_cong, masks, args, out_dir),

#                    "shipment_delayed_6h_or_more": run_task("delay", df, y_delay, masks, args, out_dir),

#                }}

#     with open(out_dir / "reports" / "evaluation_metrics_v2.json", "w", encoding="utf-8") as f:

#         json.dump(metrics, f, indent=2, default=str)



#     print_summary("congestion", metrics["targets"]["congestion_level"])

#     print_summary("delay", metrics["targets"]["shipment_delayed_6h_or_more"])

#     print(f"\nSaved models to {out_dir / 'models'} and reports to {out_dir / 'reports'}")

#     print("Reminder: synthetic data; the test period has few High cases, so read the confidence intervals.")





# if __name__ == "__main__":

#     main()









# """Train leakage-aware congestion and delay models on the synthetic Aiqon dataset."""
# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
#                              classification_report, confusion_matrix, roc_auc_score,
#                              precision_score, recall_score)
# import joblib

# ROOT = Path(__file__).resolve().parents[1]
# DATA = ROOT / 'data' / 'realistic_port_congestion_800.csv'
# MODEL_DIR = ROOT / 'models'
# REPORT_DIR = ROOT / 'reports'
# MODEL_DIR.mkdir(exist_ok=True)
# REPORT_DIR.mkdir(exist_ok=True)

# # These are outcomes or post-prediction observations and must never be features.
# LEAKAGE_COLUMNS = [
#     'observed_queue_length_vessels',
#     'observed_waiting_time_hours',
#     'observed_shipment_delay_hours',
#     'congestion_level',
#     'shipment_delayed_6h_or_more',
#     'date',
# ]
# TARGETS = ['congestion_level', 'shipment_delayed_6h_or_more']


# def make_preprocessor(X):
#     cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
#     num_cols = [c for c in X.columns if c not in cat_cols]
#     numeric = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler()),
#     ])
#     categorical = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore')),
#     ])
#     return ColumnTransformer([
#         ('numeric', numeric, num_cols),
#         ('categorical', categorical, cat_cols),
#     ])


# def chronological_masks(df):
#     dates = np.sort(df['date'].dropna().unique())
#     cut_train = int(len(dates) * 0.70)
#     cut_val = int(len(dates) * 0.85)
#     train_dates = set(dates[:cut_train])
#     val_dates = set(dates[cut_train:cut_val])
#     test_dates = set(dates[cut_val:])
#     return df['date'].isin(train_dates), df['date'].isin(val_dates), df['date'].isin(test_dates), dates, cut_train, cut_val


# def evaluate(y_true, y_pred, labels):
#     return {
#         'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
#         'balanced_accuracy': round(float(balanced_accuracy_score(y_true, y_pred)), 4),
#         'macro_f1': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
#         'weighted_f1': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
#         'labels_in_confusion_matrix': [str(x) for x in labels],
#         'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
#         'classification_report': classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
#     }


# def main():
#     df = pd.read_csv(DATA)
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     if df['date'].isna().any():
#         raise ValueError('Some dates could not be parsed. Fix date values before training.')
#     if df.duplicated().any():
#         print(f'WARNING: {int(df.duplicated().sum())} exact duplicate rows found.')

#     train_mask, val_mask, test_mask, dates, cut_train, cut_val = chronological_masks(df)
#     X = df.drop(columns=LEAKAGE_COLUMNS)
#     y_cong = df['congestion_level'].astype(str)
#     y_delay = df['shipment_delayed_6h_or_more'].astype(int)

#     # Models are trained on the earlier 70%; validation period is reserved for later tuning.
#     # The final 15% is a held-out chronological test period.
#     model_specs = {
#         'logistic_regression': LogisticRegression(max_iter=3000, class_weight='balanced'),
#         'random_forest': RandomForestClassifier(
#             n_estimators=500, min_samples_leaf=2, max_features='sqrt',
#             class_weight='balanced_subsample', random_state=42, n_jobs=-1,
#         ),
#     }
#     all_metrics = {
#         'dataset': {
#             'synthetic': True,
#             'rows': int(len(df)),
#             'columns': int(df.shape[1]),
#             'date_start': str(df['date'].min().date()),
#             'date_end': str(df['date'].max().date()),
#             'missing_values_by_column': {k: int(v) for k, v in df.isna().sum().items() if v > 0},
#             'exact_duplicate_rows': int(df.duplicated().sum()),
#             'congestion_class_counts': {str(k): int(v) for k, v in y_cong.value_counts().items()},
#             'delay_class_counts': {str(k): int(v) for k, v in y_delay.value_counts().items()},
#         },
#         'split': {
#             'method': 'chronological by unique date; no random shuffling',
#             'train_dates': [str(dates[0])[:10], str(dates[cut_train-1])[:10]],
#             'validation_dates': [str(dates[cut_train])[:10], str(dates[cut_val-1])[:10]],
#             'test_dates': [str(dates[cut_val])[:10], str(dates[-1])[:10]],
#             'train_rows': int(train_mask.sum()),
#             'validation_rows': int(val_mask.sum()),
#             'test_rows': int(test_mask.sum()),
#         },
#         'feature_columns': list(X.columns),
#         'targets': {},
#     }

#     for target_name, y in [('congestion_level', y_cong), ('shipment_delayed_6h_or_more', y_delay)]:
#         all_metrics['targets'][target_name] = {}
#         for model_name, estimator in model_specs.items():
#             pipeline = Pipeline([('preprocessor', make_preprocessor(X)), ('model', estimator)])
#             pipeline.fit(X.loc[train_mask], y.loc[train_mask])
#             pred_val = pipeline.predict(X.loc[val_mask])
#             pred_test = pipeline.predict(X.loc[test_mask])
#             classes = list(pipeline.named_steps['model'].classes_)
#             metrics = {
#                 'validation': evaluate(y.loc[val_mask], pred_val, classes),
#                 'test': evaluate(y.loc[test_mask], pred_test, classes),
#             }
#             if target_name == 'shipment_delayed_6h_or_more' and hasattr(pipeline, 'predict_proba') and 1 in classes:
#                 proba = pipeline.predict_proba(X.loc[test_mask])[:, classes.index(1)]
#                 metrics['test']['roc_auc'] = round(float(roc_auc_score(y.loc[test_mask], proba)), 4) if y.loc[test_mask].nunique() == 2 else None
#                 metrics['test']['delay_precision'] = round(float(precision_score(y.loc[test_mask], pred_test, pos_label=1, zero_division=0)), 4)
#                 metrics['test']['delay_recall'] = round(float(recall_score(y.loc[test_mask], pred_test, pos_label=1, zero_division=0)), 4)
#             all_metrics['targets'][target_name][model_name] = metrics
#             joblib.dump(pipeline, MODEL_DIR / f'{target_name}__{model_name}.joblib')

#             # Save row-level held-out predictions for review.
#             out = df.loc[test_mask, ['port_name', 'date']].copy()
#             out['actual'] = y.loc[test_mask].values
#             out['predicted'] = pred_test
#             if hasattr(pipeline, 'predict_proba'):
#                 probabilities = pipeline.predict_proba(X.loc[test_mask])
#                 out['max_class_probability'] = probabilities.max(axis=1)
#                 if target_name == 'shipment_delayed_6h_or_more' and 1 in classes:
#                     out['predicted_delay_probability'] = probabilities[:, classes.index(1)]
#             out.to_csv(REPORT_DIR / f'{target_name}__{model_name}__test_predictions.csv', index=False)

#     # A transparent data-quality summary, not a model result.
#     data_summary = {
#         'shape': [int(df.shape[0]), int(df.shape[1])],
#         'missing_values': {k: int(v) for k, v in df.isna().sum().items() if v > 0},
#         'duplicate_rows': int(df.duplicated().sum()),
#         'congestion_counts': {str(k): int(v) for k, v in y_cong.value_counts().items()},
#         'delay_counts': {str(k): int(v) for k, v in y_delay.value_counts().items()},
#         'congestion_summary': df.groupby('congestion_level')[['observed_queue_length_vessels','observed_waiting_time_hours','observed_shipment_delay_hours']].mean().round(3).to_dict(),
#     }
#     with open(REPORT_DIR / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
#         json.dump(all_metrics, f, indent=2, default=str)
#     with open(REPORT_DIR / 'data_quality_summary.json', 'w', encoding='utf-8') as f:
#         json.dump(data_summary, f, indent=2, default=str)

#     print('Chronological split:', all_metrics['split'])
#     print('Synthetic data quality:', json.dumps(data_summary, indent=2))
#     for target, models in all_metrics['targets'].items():
#         print('\nTARGET:', target)
#         for model, result in models.items():
#             print(model, 'TEST:', {k: result['test'].get(k) for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1','roc_auc','delay_precision','delay_recall'] if k in result['test']})
#     print('\nSaved models to', MODEL_DIR)
#     print('Saved reports to', REPORT_DIR)

# if __name__ == '__main__':
#     main()


"""Leakage-aware, validation-driven training for the synthetic port-congestion dataset (v2).



What changed versus train_models.py

  1. The validation period is actually used: feature set, model family and

     regularisation are chosen on validation only (threshold-free ROC-AUC).

  2. The decision threshold (High congestion / delay) is tuned on validation only.

  3. The test period is scored once, for the selected model, next to the original

     baseline (Logistic Regression, all features, default C).

  4. Per-port metrics and bootstrap 95% confidence intervals are reported, because

     the test period has very few High cases.

  5. Rolling-origin (expanding window) cross-validation gives a steadier estimate

     than a single split. Hyperparameters are fixed there, and no tuned threshold

     is applied, so it is a comparison of model families, not a nested evaluation.



The data is synthetic. Results describe this dataset, not real port operations.



Usage:

    python train_models_v2.py                       # data in ../data/, outputs in ../

    python train_models_v2.py --data path/to.csv --out some/dir

    python train_models_v2.py --refit               # also report a train+val refit

    python train_models_v2.py --beta 2              # tune thresholds to favour recall

"""

from __future__ import annotations



import argparse

import json

import warnings

from pathlib import Path



import joblib

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,

                             confusion_matrix, f1_score, fbeta_score,

                             precision_score, recall_score, roc_auc_score)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler



warnings.filterwarnings("ignore")



# Outcomes / post-prediction observations: never features.

LEAKAGE_COLUMNS = [

    "observed_queue_length_vessels",

    "observed_waiting_time_hours",

    "observed_shipment_delay_hours",

    "congestion_level",

    "shipment_delayed_6h_or_more",

    "date",

]

CALENDAR = ["month", "day_of_week", "is_weekend", "is_public_holiday"]

VESSEL_MIX = ["scheduled_container_vessels", "scheduled_bulk_vessels",

              "scheduled_tanker_vessels", "scheduled_other_vessels"]

# Candidate feature sets, expressed as columns to DROP (validation decides).

FEATURE_SETS = {

    "all": [],

    "no_calendar": CALENDAR,

    "compact": CALENDAR + VESSEL_MIX,

}

LR_C_GRID = (0.05, 0.2, 1.0)

THRESHOLD_GRID = np.round(np.arange(0.01, 1.001, 0.01), 2)

N_BOOT = 2000

SEED = 42





# --------------------------------------------------------------------------- utils

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),

                        ("scaler", StandardScaler())])

    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),

                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("numeric", numeric, num_cols),

                              ("categorical", categorical, cat_cols)])





def chronological_masks(df: pd.DataFrame):

    dates = np.sort(df["date"].unique())

    c1, c2 = int(len(dates) * 0.70), int(len(dates) * 0.85)

    return (df["date"].isin(dates[:c1]), df["date"].isin(dates[c1:c2]),

            df["date"].isin(dates[c2:]), dates, c1, c2)





def make_candidates():

    """(name, feature_set, estimator_factory) for every model/feature-set pair."""

    cands = []

    for fs in FEATURE_SETS:

        for C in LR_C_GRID:

            cands.append((f"LR|{fs}|C={C}", fs, lambda C=C: LogisticRegression(

                max_iter=3000, class_weight="balanced", C=C)))

        cands.append((f"HGB|{fs}", fs, lambda: HistGradientBoostingClassifier(

            class_weight="balanced", max_depth=3, learning_rate=0.05,

            max_iter=120, random_state=SEED)))

        cands.append((f"RF|{fs}", fs, lambda: RandomForestClassifier(

            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",

            random_state=SEED, n_jobs=-1)))

    return cands





def features_for(df: pd.DataFrame, fs: str) -> pd.DataFrame:

    return df.drop(columns=LEAKAGE_COLUMNS + FEATURE_SETS[fs])





def fit_pipeline(df, mask, fs, factory, y):

    X = features_for(df, fs)

    pipe = Pipeline([("preprocessor", make_preprocessor(X)), ("model", factory())])

    pipe.fit(X.loc[mask], y.loc[mask])

    return pipe, X





# ------------------------------------------------------------------ decisions

def decide_congestion(proba, classes, t_high):

    """Predict High when P(High) >= t_high, otherwise the likelier of Low/Moderate."""

    hi = classes.index("High")

    others = [i for i in range(len(classes)) if i != hi]

    best_other = np.array([classes[others[int(np.argmax(p[others]))]] for p in proba],

                          dtype=object)

    return np.where(proba[:, hi] >= t_high, "High", best_other)





def tune_threshold(y_true, score_fn):

    """Generic F-beta threshold tuning on validation (used for shipment delay)."""

    best = max(THRESHOLD_GRID, key=lambda t: (score_fn(t), -abs(t - 0.5)))

    return float(best)


def tune_high_recall_threshold(y_true, proba, classes, max_false_positive_rate=0.25, beta=2.0):
    """Maximize High recall on validation subject to a false-positive-rate cap.

    The constraint is computed among genuinely non-High validation rows. For a
    25% cap, no more than one quarter of non-High rows should be flagged High.
    Test data is never used for threshold selection.
    """
    y_true = np.asarray(y_true, dtype=object)
    high_truth = (y_true == "High")
    non_high = ~high_truth
    rows = []
    for threshold in THRESHOLD_GRID:
        pred = decide_congestion(proba, classes, float(threshold))
        high_pred = (pred == "High")
        recall = recall_score(high_truth, high_pred, zero_division=0)
        precision = precision_score(high_truth, high_pred, zero_division=0)
        false_positives = int((non_high & high_pred).sum())
        negatives = int(non_high.sum())
        fpr = false_positives / negatives if negatives else 0.0
        fbeta = fbeta_score(high_truth, high_pred, beta=beta, zero_division=0)
        rows.append({"threshold": float(threshold), "recall": float(recall),
                     "precision": float(precision), "fbeta": float(fbeta),
                     "false_positives": false_positives,
                     "non_high_validation_rows": negatives,
                     "false_positive_rate": float(fpr)})

    qualifying = [r for r in rows if r["false_positive_rate"] <= max_false_positive_rate]
    # There is always a no-alert operating point at threshold=1.0 unless a
    # probability is exactly 1.0; in that edge case, 1.01 ensures a feasible point.
    if not qualifying:
        pred_none = np.repeat("Low", len(y_true))
        return 1.01, {
            "max_validation_false_positive_rate": float(max_false_positive_rate),
            "validation_false_alarm_constraint_met": True,
            "validation_high_recall_at_threshold": 0.0,
            "validation_high_precision_at_threshold": 0.0,
            "validation_false_positives": 0,
            "validation_false_positive_rate": 0.0,
            "selection_policy": "fallback no-High-alert threshold to enforce false-positive-rate cap",
        }

    # First maximize recall under the false-alarm budget; then prefer precision,
    # then a higher threshold when scores are tied.
    chosen = max(qualifying, key=lambda r: (r["recall"], r["precision"], r["threshold"]))
    return chosen["threshold"], {
        "max_validation_false_positive_rate": float(max_false_positive_rate),
        "validation_false_alarm_constraint_met": chosen["false_positive_rate"] <= max_false_positive_rate,
        "validation_high_recall_at_threshold": round(chosen["recall"], 4),
        "validation_high_precision_at_threshold": round(chosen["precision"], 4),
        "validation_false_positives": chosen["false_positives"],
        "validation_non_high_rows": chosen["non_high_validation_rows"],
        "validation_false_positive_rate": round(chosen["false_positive_rate"], 4),
        "selection_policy": "maximize validation High recall subject to maximum validation false-positive rate",
    }



# -------------------------------------------------------------------- metrics

def _safe_div(a, b):

    return float(a) / b if b else float("nan")





def cong_metrics(y, pred, classes, ports=None):

    y, pred = np.asarray(y, dtype=object), np.asarray(pred, dtype=object)

    out = {

        "accuracy": round(float(accuracy_score(y, pred)), 4),

        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),

        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),

        "high_recall": round(float(recall_score(y == "High", pred == "High", zero_division=0)), 4),

        "high_precision": round(float(precision_score(y == "High", pred == "High", zero_division=0)), 4),

        "n_high": int((y == "High").sum()),

        "labels": classes,

        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),

    }

    if ports is not None:

        ports = np.asarray(ports)

        out["per_port"] = {}

        for p in sorted(set(ports)):

            m = ports == p

            out["per_port"][p] = {

                "n": int(m.sum()), "n_high": int((y[m] == "High").sum()),

                "high_recall": round(float(recall_score(y[m] == "High", pred[m] == "High", zero_division=0)), 3),

                "high_precision": round(float(precision_score(y[m] == "High", pred[m] == "High", zero_division=0)), 3),

                "accuracy": round(float(accuracy_score(y[m], pred[m])), 3),

            }

    return out





def delay_metrics(y, pred, proba=None, ports=None):

    y, pred = np.asarray(y), np.asarray(pred)

    out = {

        "accuracy": round(float(accuracy_score(y, pred)), 4),

        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),

        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),

        "delay_precision": round(float(precision_score(y, pred, zero_division=0)), 4),

        "delay_recall": round(float(recall_score(y, pred, zero_division=0)), 4),

        "n_delayed": int(y.sum()),

        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),

    }

    if proba is not None and len(set(y)) == 2:

        out["roc_auc"] = round(float(roc_auc_score(y, proba)), 4)

    if ports is not None:

        ports = np.asarray(ports)

        out["per_port"] = {}

        for p in sorted(set(ports)):

            m = ports == p

            out["per_port"][p] = {

                "n": int(m.sum()), "n_delayed": int(y[m].sum()),

                "delay_recall": round(float(recall_score(y[m], pred[m], zero_division=0)), 3),

                "delay_precision": round(float(precision_score(y[m], pred[m], zero_division=0)), 3),

            }

    return out





def bootstrap_ci(y, pred, fn, n=N_BOOT, seed=0):

    """95% percentile bootstrap CI over test rows (resamples with no positives are skipped)."""

    rng = np.random.default_rng(seed)

    y, pred = np.asarray(y, dtype=object), np.asarray(pred, dtype=object)

    vals = []

    for _ in range(n):

        i = rng.integers(0, len(y), len(y))

        v = fn(y[i], pred[i])

        if not np.isnan(v):

            vals.append(v)

    lo, hi = np.percentile(vals, [2.5, 97.5])

    return [round(float(lo), 3), round(float(hi), 3)]





def ci_block_cong(y, pred):

    return {

        "balanced_accuracy_95ci": bootstrap_ci(y, pred, lambda a, b: balanced_accuracy_score(a, b)),

        "high_recall_95ci": bootstrap_ci(

            y, pred, lambda a, b: _safe_div(((a == "High") & (b == "High")).sum(), (a == "High").sum())),

    }





def ci_block_delay(y, pred):

    y, pred = np.asarray(y, dtype=int).astype(object), np.asarray(pred, dtype=int).astype(object)

    return {

        "balanced_accuracy_95ci": bootstrap_ci(y, pred, lambda a, b: balanced_accuracy_score(a.astype(int), b.astype(int))),

        "delay_recall_95ci": bootstrap_ci(

            y, pred, lambda a, b: _safe_div(((a == 1) & (b == 1)).sum(), (a == 1).sum())),

        "delay_precision_95ci": bootstrap_ci(

            y, pred, lambda a, b: _safe_div(((a == 1) & (b == 1)).sum(), (b == 1).sum())),

    }





# ----------------------------------------------------------------- selection

def val_score(task, y_val, proba, classes):

    if task == "congestion":

        return roc_auc_score(y_val, proba, multi_class="ovr", average="macro", labels=classes)

    return roc_auc_score(y_val, proba[:, classes.index(1)])





def select_model(df, y, task, masks):

    train_mask, val_mask, _ = masks

    rows = []

    for name, fs, factory in make_candidates():

        pipe, X = fit_pipeline(df, train_mask, fs, factory, y)

        classes = list(pipe.named_steps["model"].classes_)

        proba = pipe.predict_proba(X.loc[val_mask])

        rows.append({"name": name, "feature_set": fs, "factory": factory,

                     "val_auc": float(val_score(task, y.loc[val_mask], proba, classes))})

    rows.sort(key=lambda r: r["val_auc"], reverse=True)

    return rows





# ------------------------------------------------------------- rolling CV

def rolling_cv(df, y, task, configs, n_folds=5, start=0.45):

    dates = np.sort(df["date"].unique())

    n = len(dates)

    edges = np.linspace(int(n * start), n, n_folds + 1).astype(int)

    results = {}

    for name, fs, factory in configs:

        pred = pd.Series(index=df.index, dtype=object)

        proba1 = pd.Series(index=df.index, dtype=float)

        for i in range(n_folds):

            tr = df["date"].isin(dates[:edges[i]])

            te = df["date"].isin(dates[edges[i]:edges[i + 1]])

            pipe, X = fit_pipeline(df, tr, fs, factory, y)

            pred.loc[te] = pipe.predict(X.loc[te])

            if task == "delay":

                classes = list(pipe.named_steps["model"].classes_)

                proba1.loc[te] = pipe.predict_proba(X.loc[te])[:, classes.index(1)]

        k = pred.notna()

        if task == "congestion":

            results[name] = cong_metrics(y[k], pred[k], ["High", "Low", "Moderate"],

                                         ports=df.loc[k, "port_name"])

        else:

            results[name] = delay_metrics(y[k].astype(int), pred[k].astype(int),

                                          proba1[k], ports=df.loc[k, "port_name"])

        results[name]["n_rows"] = int(k.sum())

    return results





# ------------------------------------------------------------------- main

def run_task(task, df, y, masks, args, out_dir):

    train_mask, val_mask, test_mask = masks

    ports_test = df.loc[test_mask, "port_name"].values

    ranking = select_model(df, y, task, masks)

    best = ranking[0]

    print(f"\n=== {task}: top validation candidates (ROC-AUC) ===")

    for r in ranking[:5]:

        print(f"  {r['val_auc']:.4f}  {r['name']}")



    # Fit selected config on TRAIN only; tune threshold on VALIDATION only.

    pipe, X = fit_pipeline(df, train_mask, best["feature_set"], best["factory"], y)

    classes = list(pipe.named_steps["model"].classes_)

    p_val, p_test = pipe.predict_proba(X.loc[val_mask]), pipe.predict_proba(X.loc[test_mask])



    if task == "congestion":

        yv = y.loc[val_mask].values.astype(object)

        # Safety-first congestion threshold: maximize recall within validation false-alarm cap.
        # The test set is not used to choose the threshold.
        thr, threshold_policy = tune_high_recall_threshold(
            yv, p_val, classes, max_false_positive_rate=args.max_high_fpr, beta=args.beta
        )

        pred_val = decide_congestion(p_val, classes, thr)

        pred_test = decide_congestion(p_test, classes, thr)

        yt = y.loc[test_mask].values.astype(object)

        val_m = cong_metrics(yv, pred_val, classes)

        test_m = cong_metrics(yt, pred_test, classes, ports=ports_test)

        test_m.update(ci_block_cong(yt, pred_test))

        test_m["decision_rule"] = f"High if P(High) >= {thr}, else likelier of Low/Moderate"
        test_m["threshold_policy"] = threshold_policy
        val_m["threshold_policy"] = threshold_policy

    else:

        yv = y.loc[val_mask].values.astype(int)

        pv, pt = p_val[:, classes.index(1)], p_test[:, classes.index(1)]

        # Tune delay threshold separately to favour recall when --delay-beta > 1.
        thr = tune_threshold(yv, lambda t: fbeta_score(
            yv,
            (pv >= t).astype(int),
            beta=args.delay_beta,
            zero_division=0
        ))

        pred_val, pred_test = (pv >= thr).astype(int), (pt >= thr).astype(int)

        yt = y.loc[test_mask].values.astype(int)

        val_m = delay_metrics(yv, pred_val, pv)

        test_m = delay_metrics(yt, pred_test, pt, ports=ports_test)

        test_m.update(ci_block_delay(yt, pred_test))

        test_m["decision_rule"] = f"Delayed if P(delay) >= {thr}"

        # Also show the default 0.5 operating point for transparency.

        test_m["at_threshold_0.5"] = delay_metrics(yt, (pt >= 0.5).astype(int), pt)



    # Original baseline for a like-for-like comparison on the same test rows.

    base_pipe, Xb = fit_pipeline(df, train_mask, "all", lambda: LogisticRegression(

        max_iter=3000, class_weight="balanced"), y)

    pb = base_pipe.predict(Xb.loc[test_mask])

    if task == "congestion":

        baseline = cong_metrics(yt, pb.astype(object), classes, ports=ports_test)

    else:

        baseline = delay_metrics(yt, pb.astype(int),

                                 base_pipe.predict_proba(Xb.loc[test_mask])[:, 1], ports=ports_test)



    result = {

        "selected": best["name"], "threshold": thr,

        "validation_ranking": [{"name": r["name"], "val_auc": round(r["val_auc"], 4)} for r in ranking],

        "validation_metrics": val_m, "test_metrics": test_m,

        "baseline_original_lr_test": baseline,

    }



    if args.refit:

        trval = train_mask | val_mask

        pipe2, X2 = fit_pipeline(df, trval, best["feature_set"], best["factory"], y)

        p2 = pipe2.predict_proba(X2.loc[test_mask])

        if task == "congestion":

            pr2 = decide_congestion(p2, classes, thr)

            result["test_metrics_refit_train_plus_val"] = cong_metrics(yt, pr2, classes, ports=ports_test)

        else:

            q2 = p2[:, classes.index(1)]

            result["test_metrics_refit_train_plus_val"] = delay_metrics(yt, (q2 >= thr).astype(int), q2, ports=ports_test)

        result["refit_note"] = ("Hyperparameters and threshold come from the train-only fit; "

                                "the refit model's probabilities may be calibrated slightly differently.")



    # Rolling CV: baseline vs selected vs other families (fixed hyperparameters).

    cv_cfgs = [("baseline_LR|all|C=1.0", "all", lambda: LogisticRegression(

                   max_iter=3000, class_weight="balanced")),

               (best["name"] + " (selected)", best["feature_set"], best["factory"])]

    for r in ranking:

        fam = r["name"].split("|")[0]

        if fam in ("HGB", "RF") and not any(c[0].startswith(fam) for c in cv_cfgs):

            cv_cfgs.append((r["name"], r["feature_set"], r["factory"]))

    result["rolling_cv"] = rolling_cv(df, y, task, cv_cfgs)



    # Save artefacts.

    joblib.dump({"pipeline": pipe, "classes": classes, "threshold": thr,

                 "config": best["name"], "feature_columns": list(X.columns)},

                out_dir / "models" / f"{task}__v2_selected.joblib")

    pred_df = df.loc[test_mask, ["port_name", "date"]].copy()

    pred_df["actual"] = yt

    pred_df["predicted"] = pred_test

    for j, c in enumerate(classes):

        pred_df[f"proba_{c}"] = p_test[:, j]

    pred_df.to_csv(out_dir / "reports" / f"{task}__v2_test_predictions.csv", index=False)

    return result





def print_summary(task, res):

    t, b = res["test_metrics"], res["baseline_original_lr_test"]

    print(f"\n##### {task.upper()} -- selected: {res['selected']}  (threshold {res['threshold']})")

    if task == "congestion":

        keys = ["accuracy", "balanced_accuracy", "macro_f1", "high_recall", "high_precision"]

    else:

        keys = ["accuracy", "balanced_accuracy", "macro_f1", "delay_recall", "delay_precision", "roc_auc"]

    print(f"{'metric':20s} {'baseline LR':>12s} {'selected':>10s}")

    for k in keys:

        print(f"{k:20s} {b.get(k, float('nan')):12.3f} {t.get(k, float('nan')):10.3f}")

    ci_keys = [k for k in t if k.endswith("_95ci")]

    for k in ci_keys:

        print(f"  {k}: {t[k]}")

    print("  per-port (test):")

    for p, v in t["per_port"].items():

        print("   ", p, v)

    if "test_metrics_refit_train_plus_val" in res:

        r = res["test_metrics_refit_train_plus_val"]

        print("  refit on train+val, same threshold (test):",

              {k: r[k] for k in keys if k in r})

    print("  rolling-origin CV (pooled over future folds):")

    for name, m in res["rolling_cv"].items():

        if task == "congestion":

            print(f"    {name:34s} bal={m['balanced_accuracy']:.3f} mF1={m['macro_f1']:.3f} "

                  f"HighRec={m['high_recall']:.2f} HighPrec={m['high_precision']:.2f} "

                  f"(n_high={m['n_high']})")

            print("       High recall per port:",

                  {p: v["high_recall"] for p, v in m["per_port"].items()})

        else:

            print(f"    {name:34s} bal={m['balanced_accuracy']:.3f} AUC={m.get('roc_auc', float('nan')):.3f} "

                  f"DelayRec={m['delay_recall']:.2f} DelayPrec={m['delay_precision']:.2f}")





def main():

    root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default=str(root / "data" / "realistic_port_congestion_800.csv"))

    ap.add_argument("--out", default=str(root))

    ap.add_argument("--refit", action="store_true", help="also report a train+val refit on test")

    ap.add_argument("--beta", type=float, default=2.0,
                    help="F-beta used as a tie-breaker for High-congestion threshold selection; "
                         "higher values favour recall (more false alarms may result)")
    ap.add_argument("--max-high-fpr", type=float, default=0.25,
                    help="Maximum allowed false-positive rate among non-High validation rows "
                         "for High-congestion alerts (default: 0.25). Threshold selection "
                         "maximizes High recall within this false-alarm budget.")

    ap.add_argument(

        "--delay-beta",

        type=float,

        default=2.0,

        help="F-beta for shipment-delay threshold tuning; higher values favour recall"

    )

    args = ap.parse_args()
    if not 0.0 <= args.max_high_fpr <= 1.0:
        ap.error("--max-high-fpr must be in the interval [0, 1].")
    if args.beta <= 0 or args.delay_beta <= 0:
        ap.error("--beta and --delay-beta must be greater than zero")

    out_dir = Path(args.out)

    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    (out_dir / "reports").mkdir(parents=True, exist_ok=True)



    df = pd.read_csv(args.data)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():

        raise ValueError("Some dates could not be parsed.")

    df = df.sort_values(["date", "port_name"]).reset_index(drop=True)

    train_mask, val_mask, test_mask, dates, c1, c2 = chronological_masks(df)

    masks = (train_mask, val_mask, test_mask)



    y_cong = df["congestion_level"].astype(str)

    y_delay = df["shipment_delayed_6h_or_more"].astype(int)



    split = {

        "train": [str(dates[0])[:10], str(dates[c1 - 1])[:10], int(train_mask.sum())],

        "validation": [str(dates[c1])[:10], str(dates[c2 - 1])[:10], int(val_mask.sum())],

        "test": [str(dates[c2])[:10], str(dates[-1])[:10], int(test_mask.sum())],

        "High_counts": {k: int((y_cong[m] == "High").sum()) for k, m in

                        [("train", train_mask), ("validation", val_mask), ("test", test_mask)]},

    }

    print("Split [start, end, rows]:", json.dumps(split))



    metrics = {"dataset": {"synthetic": True, "rows": int(len(df))}, "split": split,

               "feature_sets_dropped_columns": FEATURE_SETS,

               "targets": {

                   "congestion_level": run_task("congestion", df, y_cong, masks, args, out_dir),

                   "shipment_delayed_6h_or_more": run_task("delay", df, y_delay, masks, args, out_dir),

               }}

    with open(out_dir / "reports" / "evaluation_metrics_v2.json", "w", encoding="utf-8") as f:

        json.dump(metrics, f, indent=2, default=str)



    print_summary("congestion", metrics["targets"]["congestion_level"])

    print_summary("delay", metrics["targets"]["shipment_delayed_6h_or_more"])

    print(f"\nSaved models to {out_dir / 'models'} and reports to {out_dir / 'reports'}")

    print("Reminder: synthetic data; the test period has few High cases, so read the confidence intervals.")





if __name__ == "__main__":

    main()

