import os
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def measure_peak_memory_mb(func, *args, **kwargs):
    """Measures peak memory RSS usage delta in MB during function execution."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    result = func(*args, **kwargs)

    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before
    return result, max(mem_delta, 0.0)


def benchmark_training(model_cls, kwargs, X_train, y_train):
    """Task 3: Warm up once, repeat 5 times, return trained model, median time, and peak memory delta."""
   
    warmup_model = model_cls(**kwargs)
    warmup_model.fit(X_train, y_train)

    times = []
    mem_deltas = []
    fitted_model = None

   
    for _ in range(5):
        m = model_cls(**kwargs)
        start = time.perf_counter()
        _, peak_mem = measure_peak_memory_mb(m.fit, X_train, y_train)
        end = time.perf_counter()

        times.append(end - start)
        mem_deltas.append(peak_mem)
        fitted_model = m

    return fitted_model, np.median(times), np.median(mem_deltas)


def benchmark_inference(model, X_test):
    """Task 3: Repeat single-sample inference 100 times, return median latency (ms) and peak memory."""
    single_sample = X_test[:1]


    model.predict(single_sample)

    latencies_ms = []
    mem_deltas = []


    for _ in range(100):
        start = time.perf_counter()
        _, peak_mem = measure_peak_memory_mb(model.predict, single_sample)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)
        mem_deltas.append(peak_mem)

    return np.median(latencies_ms), np.median(mem_deltas)


def main():
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    models_config = {
        "Logistic Regression": {
            "class": LogisticRegression,
            "kwargs": {"max_iter": 1000, "random_state": SEED},
            "file": results_dir / "model_logistic.joblib",
        },
        "Random Forest": {
            "class": RandomForestClassifier,
            "kwargs": {"n_estimators": 100, "random_state": SEED},
            "file": results_dir / "model_rf.joblib",
        },
    }

    acc_records = []
    system_records = []

    for name, cfg in models_config.items():
        model, med_train_sec, peak_train_mem = benchmark_training(
            cfg["class"], cfg["kwargs"], X_train, y_train
        )


        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_records.append({"Model": name, "Accuracy": f"{acc:.4f}"})

        joblib.dump(model, cfg["file"])
        model_size_bytes = cfg["file"].stat().st_size
        model_size_kb = model_size_bytes / 1024.0

     
        med_infer_ms, peak_infer_mem = benchmark_inference(model, X_test)

        system_records.append(
            {
                "Model": name,
                "Training Time (s)": round(med_train_sec, 4),
                "Inference Latency (ms)": round(med_infer_ms, 4),
                "Model Size (Bytes)": model_size_bytes,
                "Model Size (KB)": round(model_size_kb, 2),
                "Peak Train Mem Delta (MB)": round(peak_train_mem, 4),
                "Peak Infer Mem Delta (MB)": round(peak_infer_mem, 4),
            }
        )

  
    df_acc = pd.DataFrame(acc_records)
    df_acc.to_csv(results_dir / "baseline_accuracy.csv", index=False)
    print(f"Saved Task 2 accuracy results to {results_dir / 'baseline_accuracy.csv'}")

    df_sys = pd.DataFrame(system_records)
    df_sys.to_csv(results_dir / "system_metrics.csv", index=False)
    print(f"Saved Task 3 system metrics to {results_dir / 'system_metrics.csv'}")


if __name__ == "__main__":
    main()