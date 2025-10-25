import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from modules.evaluation import evaluate_svm
from modules.load import load_5d_classification_data, split_5d_data


def _report_linear_weights(best_model, feature_names=None):
    svc = best_model.named_steps["svc"]
    if getattr(svc, "kernel", None) != "linear":
        return
    coefs = svc.coef_.ravel()
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(coefs.size)]
    order = np.argsort(np.abs(coefs))[::-1]
    print("\nFeature weights (linear SVM, sorted by |weight|):")
    for idx in order:
        print(f"{feature_names[idx]:>4}: {coefs[idx]: .4f}")


def tune_svm(X_train, y_train, seed=507, cv_folds=5, refit_metric="f1"):
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=seed))
    ])

    param_grid = [
        {"svc__kernel": ["linear"],
         "svc__C": np.logspace(-3, 3, 7),
         "svc__class_weight": [None, "balanced"]},

        {"svc__kernel": ["rbf"],
         "svc__C": np.logspace(-3, 3, 7),
         "svc__gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 9)),
         "svc__class_weight": [None, "balanced"]},
    ]

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    min_class = np.bincount(y_train).min()
    cv_folds = min(cv_folds, max(2, min_class))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid


def train_svm(X, y, refit_metric="f1", seed=507):
    X_train, X_test, y_train, y_test = split_5d_data(X, y, test_size=0.25, seed=seed)

    grid = tune_svm(X_train, y_train, seed=seed, cv_folds=5, refit_metric=refit_metric)

    results = pd.DataFrame(grid.cv_results_)
    for column in ["param_svc__kernel", "param_svc__C", "param_svc__gamma",
                "mean_test_accuracy", "mean_test_precision", "mean_test_recall",
                "mean_test_f1", "mean_test_roc_auc", "rank_test_" + grid.refit]:
        if column not in results.columns:
            results[column] = np.nan

    results["param_svc__gamma"] = results["param_svc__gamma"].astype(object)
    results.loc[results["param_svc__kernel"] == "linear", "param_svc__gamma"] = "-"

    print("\nTop configurations by refit metric (lower rank is better):")
    display_df = results.sort_values("rank_test_" + grid.refit).head(10)[
        ["param_svc__kernel", "param_svc__C", "param_svc__gamma",
         "mean_test_accuracy", "mean_test_precision", "mean_test_recall",
         "mean_test_f1", "mean_test_roc_auc", "rank_test_" + grid.refit]
    ]
    print(display_df.to_string(index=False))

    print("\nBest params:", grid.best_params_)
    print(f"Best CV {grid.refit}: {grid.best_score_:.3f}")

    best_model = grid.best_estimator_
    acc, prec, rec, f1, auc = evaluate_svm(best_model, X_test, y_test, title_prefix="Best SVM")
    _report_linear_weights(best_model, feature_names=["X1", "X2", "X3", "X4", "X5"])

    counts_total = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    counts_train = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    counts_test = {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}

    return grid, best_model, acc, prec, rec, f1, auc, counts_total, counts_train, counts_test
