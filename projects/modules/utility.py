import os
import time
import joblib
import argparse
import uuid
from datetime import datetime
from pathlib import Path


def save_model(pipe, features, model):
    models_path = os.path.join(os.getcwd(), "models")

    # Always ensure directory exists
    os.makedirs(models_path, exist_ok=True)

    pipe_path = os.path.join(models_path, "preprocessing_pipe.pkl")
    feat_path = os.path.join(models_path, "selected_features.pkl")
    model_path = os.path.join(models_path, "xgb_model.pkl")

    # Save (joblib overwrites by default)
    joblib.dump(pipe, pipe_path)
    joblib.dump(features, feat_path)
    joblib.dump(model, model_path)

    # Verify overwrite by modified time
    print(f"Model saved successfully at {models_path}")
    print("Model last modified:", time.ctime(os.path.getmtime(model_path)))


def load_model():
    models_path = os.path.join(os.getcwd(), 'models')

    if os.path.exists(models_path):
        pipe = joblib.load(os.path.join(models_path, "preprocessing_pipe.pkl"))
        selected_features = joblib.load(os.path.join(models_path, "selected_features.pkl"))
        model = joblib.load(os.path.join(models_path, "xgb_model.pkl"))
        return pipe, selected_features, model
    else:
        raise OSError('Directory not found. Model directory missing.')


def find_project_root(marker=".project_root"):
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def create_run_id():
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def prepare_report_dir(report_dir):
    os.makedirs(report_dir, exist_ok=True)


def save_classification_report(
    report_text,
    run_id,
    report_dir="reports",
    metadata=None
):
    prepare_report_dir(report_dir)

    path = os.path.join(
        report_dir,
        f"classification_report_{run_id}.txt"
    )

    with open(path, "w") as f:
        f.write("XGBoost Classification Report\n")
        f.write(f"Run ID: {run_id}\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

        f.write("\n")
        f.write(report_text)

    return path

