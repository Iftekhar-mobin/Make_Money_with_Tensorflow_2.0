import os
import joblib
import pickle
from pathlib import Path


def save_selected_features(selected_features, file_name="selected_features.pkl"):
    dir_path = "models"
    os.makedirs(dir_path, exist_ok=True)

    path = os.path.join(dir_path, file_name)
    with open(path, "wb") as f:
        pickle.dump(list(selected_features), f)


def load_selected_features(file_name="selected_features.pkl"):
    dir_path = "models"
    path = os.path.join(dir_path, file_name)

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


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



