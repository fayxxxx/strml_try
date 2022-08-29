import numpy as np
import lightgbm as lgbm
import optuna  # pip install optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
def objective(trial, X, y, cat_col):
    # 参数网格
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.11, step = 0.5),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50, step = 10),
        "max_depth": trial.suggest_int("max_depth", 4, 9, step = 2),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1.0, step = 0.1),
        "bagging_freq": trial.suggest_int("bagging_freq", 2, 5, step = 1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.1, step = 0.2),
        "random_state": 0,
    }
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # LGBM建模
        model = lgbm.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
            early_stopping_rounds=100,
            categorical_feature=cat_col
        )
        # 模型预测
        preds = model.predict_proba(X_test)
        cv_scores[idx] = roc_auc_score(y_test, preds[:,1])

    return np.mean(cv_scores)