import sys
import warnings
from pathlib import Path

# Silence deprecation/future warnings from PyTorch/pynvml
warnings.filterwarnings("ignore")


def main():
    # Resolve directory paths relative to this script
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "versions.txt"

    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "torchinfo",
        "thop",
        "onnx",
        "onnxruntime",
        "mlflow",
        "memory_profiler",
        "psutil",
        "codecarbon",
        "fastapi",
        "uvicorn",
        "pytest",
        "httpx",
        "locust",
        "requests",
        "pyarrow",
        "joblib",
        "tqdm",
    ]

    version_lines = [f"python=={sys.version.split()[0]}"]

    for pkg_name in packages:
        # Import name mapping for sklearn and memory_profiler
        module_import_name = pkg_name
        if pkg_name == "scikit-learn":
            module_import_name = "sklearn"
        elif pkg_name == "memory_profiler":
            module_import_name = "memory_profiler"

        try:
            mod = __import__(module_import_name)
            ver = getattr(mod, "__version__", None)

            if ver is None:
                import importlib.metadata

                ver = importlib.metadata.version(pkg_name)

            version_lines.append(f"{pkg_name}=={ver}")
        except Exception as e:
            version_lines.append(f"{pkg_name}==error_loading ({e})")

    content = "\n".join(version_lines) + "\n"

    # Write directly to lab01/results/versions.txt
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully wrote {len(version_lines)} package versions to {output_file}")


if __name__ == "__main__":
    main()