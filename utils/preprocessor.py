"""
preprocessor.py
---------------
Preprocessing pipeline for classification only.

Steps:
  1. Drop columns where >60% of values are missing (too sparse to impute)
  2. Drop rows where the TARGET is missing (can't train without a label)
  3. Impute missing values (median for numeric, mode for categorical)
     — replaces the old dropna() which wiped entire datasets with any NaN
  4. Encode categorical feature columns
  5. Label-encode target column
  6. Train / Val / Test split (70 / 15 / 15)
  7. Cap outliers using IQR
  8. StandardScaler on features
  9. Optional SMOTE for class imbalance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder


def _cap_outliers(X: np.ndarray) -> np.ndarray:
    """Cap extreme outlier values using IQR (3× rule) per feature."""
    X = X.copy()
    for col in range(X.shape[1]):
        Q1  = np.percentile(X[:, col], 25)
        Q3  = np.percentile(X[:, col], 75)
        IQR = Q3 - Q1
        X[:, col] = np.clip(X[:, col], Q1 - 3 * IQR, Q3 + 3 * IQR)
    return X


def preprocess(df: pd.DataFrame, target_col: str, problem_type: str = 'classification',
               use_smote: bool = False):
    """
    Full preprocessing pipeline (classification only).

    Args:
        df:           raw input DataFrame
        target_col:   name of the column to predict
        problem_type: ignored — always treated as classification
        use_smote:    whether to apply SMOTE oversampling
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, feature_names
    """
    df = df.copy()

    # 1. Drop columns where more than 60% of values are missing
    thresh = int(0.4 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    # 2. Drop rows where the target itself is missing — can't train without a label
    df = df.dropna(subset=[target_col])

    if len(df) == 0:
        raise ValueError(
            "Dataset is empty after removing rows with missing target values. "
            "Please check that your target column is correct."
        )

    # 3. Split features / target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    feature_names = list(X.columns)

    # 4. Impute missing values — median for numeric, mode for categorical
    #    This replaces dropna() so datasets with partial NaNs still work.
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['object', 'category'] or X[col].nunique() < 20:
                fill_val = X[col].mode(dropna=True)
                X[col] = X[col].fillna(fill_val[0] if len(fill_val) > 0 else 'missing')
            else:
                X[col] = X[col].fillna(X[col].median())

    # 5. Encode categorical feature columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        if X[col].nunique() > 50:
            freq_map = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq_map).fillna(0)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 6. Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y.astype(str))
    num_classes = len(np.unique(y))

    X = X.values.astype(float)

    # 7. Train / Val / Test split  (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # 8. Cap outliers (fit on train only)
    X_train = _cap_outliers(X_train)
    X_val   = _cap_outliers(X_val)
    X_test  = _cap_outliers(X_test)

    # 9. Scale features (fit on train only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # 10. Optional SMOTE
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        except ImportError:
            print("[Preprocessor] imbalanced-learn not installed. Skipping SMOTE.")

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes, feature_names