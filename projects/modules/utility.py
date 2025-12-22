import os
import joblib
from pathlib import Path


def save_model(pipe, features, model):
    models_path = os.path.join(os.getcwd(), 'models')

    if os.path.exists(models_path):
        joblib.dump(pipe, os.path.join(models_path, "preprocessing_pipe.pkl"))
        joblib.dump(features, os.path.join(models_path, "selected_features.pkl"))
        joblib.dump(model, os.path.join(models_path, "xgb_model.pkl"))
        print(f"Model saved successfully at {models_path}.")

    else:
        raise OSError('Directory not found. Please download properly')


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



