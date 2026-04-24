"""
comparator.py
-------------
Trains classical ML baseline classifiers and evaluates them
against the NAS-discovered neural network.
"""

import time
import numpy as np

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier, GradientBoostingClassifier,
                                   ExtraTreesClassifier)
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.metrics       import accuracy_score, f1_score, roc_auc_score


def _get_classification_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)":           SVC(kernel='rbf', probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Naive Bayes":         GaussianNB(),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric='logloss', verbosity=0
        )
    except ImportError:
        pass
    return models


def run_classification_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    num_classes: int
) -> list:
    """
    Train and evaluate all classification baselines.

    Returns list of result dicts:
        model, accuracy, f1, roc_auc, predict_time, params, status
    """
    results   = []
    is_binary = (num_classes == 2)

    for name, clf in _get_classification_models().items():
        try:
            clf.fit(X_train, y_train)
            # Measure prediction time only — fair comparison with NAS inference time
            t0      = time.time()
            y_pred  = clf.predict(X_test)
            elapsed = round(time.time() - t0, 6)

            acc = round(accuracy_score(y_test, y_pred), 5)
            f1  = round(f1_score(y_test, y_pred,
                                 average='binary' if is_binary else 'weighted',
                                 zero_division=0), 5)

            roc = None
            if is_binary and hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(X_test)[:, 1]
                roc    = round(roc_auc_score(y_test, y_prob), 5)

            # Estimate parameter count for sklearn models
            try:
                if hasattr(clf, 'coef_'):
                    params = int(clf.coef_.size + (clf.intercept_.size if hasattr(clf, 'intercept_') else 0))
                elif hasattr(clf, 'estimators_'):
                    params = sum(
                        e.tree_.node_count for e in clf.estimators_
                        if hasattr(e, 'tree_')
                    )
                elif hasattr(clf, 'tree_'):
                    params = int(clf.tree_.node_count)
                elif hasattr(clf, 'support_vectors_'):
                    params = int(clf.support_vectors_.size)
                else:
                    params = None
            except Exception:
                params = None

            results.append({
                'model':      name,
                'accuracy':   acc,
                'f1':         f1,
                'roc_auc':    roc,
                'predict_time': elapsed,
                'params':     params,
                'status':     'ok'
            })

        except Exception as e:
            results.append({
                'model':      name,
                'accuracy':   None,
                'f1':         None,
                'roc_auc':    None,
                'predict_time': None,
                'params':     None,
                'status':     f'error: {str(e)[:60]}'
            })

    return results


def evaluate_nas_model_classification(model, X_test, y_test, num_classes):
    """
    Compute classification metrics for the final Keras NAS model.
    Returns dict: accuracy, f1, roc_auc
    """
    is_binary = (num_classes == 2)

    if is_binary:
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        roc    = round(roc_auc_score(y_test, y_prob), 5)
    else:
        y_probs = model.predict(X_test, verbose=0)
        y_pred  = np.argmax(y_probs, axis=1)
        roc     = None

    acc = round(accuracy_score(y_test, y_pred), 5)
    f1  = round(f1_score(y_test, y_pred,
                         average='binary' if is_binary else 'weighted',
                         zero_division=0), 5)

    return {'accuracy': acc, 'f1': f1, 'roc_auc': roc}