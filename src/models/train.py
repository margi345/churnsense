import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data.loader import load_config, load_processed
from src.models.evaluate import evaluate, print_metrics


# Load config 
config = load_config()


# Prepare data 
def prepare_data(df: pd.DataFrame = None):
    if df is None:
        df = load_processed()

    target = "Churn"

    if target not in df.columns:
        raise ValueError(
            "Churn column not found. Run: python -m src.data.features"
        )

    X = df.drop(columns=[target])
    y = df[target]

    print(f"Features: {X.shape[1]} columns")
    print(f"Samples:  {len(y)} rows")
    print(f"Churn rate: {y.mean():.1%}")
    return X, y

# Baseline: Logistic Regression 
def train_baseline(X, y) -> dict:
    print("\nTraining baseline (Logistic Regression)...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config["data"]["random_state"]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=config["data"]["random_state"])
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    print(f"  Baseline AUC-ROC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    model.fit(X, y)
    return {"model": model, "auc_roc": scores.mean(), "name": "LogisticRegression"}


# XGBoost with Optuna 
def train_xgboost(X, y) -> dict:
    print("\nTraining XGBoost with Optuna...")

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": scale_pos_weight,
            "random_state":     config["data"]["random_state"],
            "eval_metric":      "auc",
            "verbosity":        0,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=config["model"]["cv_folds"],
                             shuffle=True,
                             random_state=config["data"]["random_state"])
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["model"]["optuna_trials"])

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = config["data"]["random_state"]
    best_params["verbosity"] = 0

    model = XGBClassifier(**best_params)
    model.fit(X, y)

    print(f"  Best AUC-ROC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return {"model": model, "auc_roc": study.best_value,
            "params": best_params, "name": "XGBoost"}


# LightGBM with Optuna
def train_lightgbm(X, y) -> dict:
    print("\nTraining LightGBM with Optuna...")

    class_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": class_weight,
            "random_state":     config["data"]["random_state"],
            "verbose":         -1,
        }
        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=config["model"]["cv_folds"],
                             shuffle=True,
                             random_state=config["data"]["random_state"])
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["model"]["optuna_trials"])

    best_params = study.best_params
    best_params["scale_pos_weight"] = class_weight
    best_params["random_state"] = config["data"]["random_state"]
    best_params["verbose"] = -1

    model = LGBMClassifier(**best_params)
    model.fit(X, y)

    print(f"  Best AUC-ROC: {study.best_value:.4f}")
    return {"model": model, "auc_roc": study.best_value,
            "params": best_params, "name": "LightGBM"}


# Main training pipeline
def train():
    mlflow.set_experiment(config["model"]["experiment_name"])

    X, y = prepare_data()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    results = []

    # Train all models
    for train_fn in [train_baseline, train_xgboost, train_lightgbm]:
        result = train_fn(X_train, y_train)
        model  = result["model"]
        name   = result["name"]

        # Evaluate on test set
        y_prob   = model.predict_proba(X_test)[:, 1]
        metrics  = evaluate(y_test.values, y_prob)

        # Log to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params(result.get("params", {}))
            mlflow.log_metrics({
                "auc_roc":   metrics["auc_roc"],
                "auc_pr":    metrics["auc_pr"],
                "f1_score":  metrics["f1_score"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
            })
            mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({
            "name":    name,
            "model":   model,
            "metrics": metrics,
            "auc_roc": metrics["auc_roc"]
        })

        print_metrics(metrics)

    # Pick best model
    best = max(results, key=lambda x: x["auc_roc"])
    print(f"\nBest model: {best['name']} with AUC-ROC: {best['auc_roc']:.4f}")

    # Save best model
    import joblib
    model_path = Path(__file__).resolve().parents[2] / "models"
    model_path.mkdir(exist_ok=True)
    joblib.dump(best["model"], model_path / "best_model.joblib")
    joblib.dump(X.columns.tolist(), model_path / "feature_names.joblib")
    print(f"Saved best model to models/best_model.joblib")

    return best


# Run directly 
if __name__ == "__main__":
    train()