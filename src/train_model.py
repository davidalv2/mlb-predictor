"""
train_model.py
--------------
Trains XGBoost with Optuna hyperparameter optimization.
Evaluates on held-out 2025 season data.

Usage:
    python src/train_model.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5),
        "random_state": 42, "eval_metric": "auc", "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def train(n_trials: int = 100):
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    print(f"\nRunning Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest ROC-AUC : {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")

    model = XGBClassifier(
        **study.best_params,
        random_state=42, eval_metric="auc", verbosity=0
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, preds)
    print(f"\n{'='*40}")
    print(f"  Final ROC-AUC : {auc:.4f}")
    print(f"{'='*40}")
    print(classification_report(
        y_test, (preds > 0.5).astype(int),
        target_names=["Away win", "Home win"]
    ))

    print("\nTop 10 most important features:")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.nlargest(10).round(4).to_string())

    joblib.dump(model,                    MODELS_DIR / "xgb_model.joblib")
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "feature_names.joblib")
    print(f"\nModel saved to {MODELS_DIR}/xgb_model.joblib")

    return model, auc


if __name__ == "__main__":
    train(n_trials=100)
