import os
import hashlib
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)

def main():
    # Ensure results directory exists relative to lab04 or current working directory
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)

    print("--- 1. Ingestion ---")
    try:
        raw_data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    except Exception:
        raw_data = fetch_openml(name='adult', version=2, as_frame=True, parser='classic')
        
    df = raw_data.frame
    row_count, col_count = df.shape
    print(f"Raw Data Loaded Successfully: {row_count} rows, {col_count} columns")

    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("--- 2. Validation ---")
    validation_logs = []
    missing_counts = (df.astype(str) == '?').sum()
    validation_logs.append("Missing ('?') Counts:\n" + missing_counts.to_string())

    range_checks = {
        'age': (17, 90),
        'hours-per-week': (1, 99),
        'capital-gain': (0, 99999),
        'capital-loss': (0, 4356)
    }
    for col, (min_v, max_v) in range_checks.items():
        violations = df[(df[col] < min_v) | (df[col] > max_v)]
        validation_logs.append(f"{col} range violations: {len(violations)}")

    val_report_path = os.path.join(results_dir, "validation_report.txt")
    with open(val_report_path, "w") as f:
        f.write("\n".join(validation_logs))

    print("--- 3. Cleaning & Transformation ---")
    target_col = 'class' if 'class' in df.columns else 'income'
    df[target_col] = df[target_col].astype(str).str.strip().str.rstrip('.')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    train_df = pd.concat([X_train, y_train], axis=1).drop_duplicates()
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_train = X_train.replace('?', np.nan)
    X_test = X_test.replace('?', np.nan)

    quantiles = X_train[num_cols].quantile([0.01, 0.99])
    for col in num_cols:
        X_train[col] = X_train[col].clip(quantiles.loc[0.01, col], quantiles.loc[0.99, col])
        X_test[col] = X_test[col].clip(quantiles.loc[0.01, col], quantiles.loc[0.99, col])

    print("--- 4. Feature Engineering ---")
    for dataset in [X_train, X_test]:
        dataset['capital_net'] = dataset['capital-gain'] - dataset['capital-loss']
        safe_age_diff = np.maximum(dataset['age'] - 16, 1)
        dataset['hours_per_age'] = dataset['hours-per-week'] / safe_age_diff

    num_cols_ext = num_cols + ['capital_net', 'hours_per_age']

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols_ext),
        ('cat', cat_transformer, cat_cols)
    ])
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    joblib.dump(preprocessor, os.path.join(results_dir, "preprocessor.joblib"))

    print("--- 5. Duplicate Conflict Audit & Baseline Evaluation ---")
    feature_cols = [c for c in df.columns if c != target_col]
    sub_df = df[feature_cols].astype(str).copy()
    sub_df['target'] = df[target_col]
    
    duplicates = sub_df[sub_df.duplicated(subset=feature_cols, keep=False)]
    if not duplicates.empty:
        conflict_count = int(duplicates.groupby(feature_cols, observed=True)['target'].nunique().gt(1).sum())
    else:
        conflict_count = 0
    print(f"Duplicate feature rows with conflicting labels: {conflict_count}")

    print("Training baseline model...")
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_proc, y_train)
    preds = clf.predict(X_test_proc)
    
    pos_label_val = '>50K' if '>50K' in y_test.values else y_test.iloc[0]
    metrics_df = pd.DataFrame([{
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "F1_Score": round(f1_score(y_test, preds, pos_label=pos_label_val), 4)
    }])
    metrics_df.to_csv(os.path.join(results_dir, "baseline_metrics.csv"), index=False)

    sha256 = hashlib.sha256(X_train_proc.tobytes()).hexdigest()
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump({'sha256': sha256, 'seed': SEED}, f)
    
    print(f"Pipeline executed successfully! Results saved to: {results_dir}")
    print(f"SHA256 Hash: {sha256}")

if __name__ == "__main__":
    main()