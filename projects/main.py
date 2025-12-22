#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import argparse

from modules.dataset_loader import load_dataset, load_test_dataset
from modules.preprocessing import (
    rename_col,
    handling_nan_after_feature_generate,
    prepare_dataset_for_model
)
from modules.chart import generate_signal_plot
from modules.signal_label_processing import (
    prepare_signal,
    visualize_dataset
)
from modules.feature_generate import extract_all_features, extract_fast_features
from modules.feature_selection import select_best_features
from modules.models import (
    xgbmodel,
    xgbmodel_adasyn,
    xgbmodel_kfold,
    xgbmodel_comparison_with_adasyn_smote,
    predict_with_new_dataset
)
from modules.simulator import run_backtesting_simulator
from modules.utility import (
    load_model,
    save_model,
    find_project_root
)


# ---------------------------
# VISUALIZATION
# ---------------------------


class SignalMLPipeline:
    """
    A full end-to-end ML pipeline for generating signals,
    feature extraction, feature selection and model training/testing.
    """

    def __init__(self, data_dir_, file_name_, test_file_path_, n_features=20):

        self.y = None
        self.X = None
        self.data_dir = data_dir_
        self.file_name = file_name_
        self.test_file_path = test_file_path_
        self.n_features = n_features

        # Will be filled during pipeline
        self.raw_data = None
        self.dataset = None
        self.df_features = None
        self.selected_features = None
        self.pipe = None
        self.model = None

        # step map (int → function)
        self.step_functions = {
            1: ("load", self.load_and_prepare_raw_data),
            2: ("label", self.generate_labels),
            3: ("simulation", self._simulation),
            4: ("visualize", self.visualize_current_dataset),  # ← moved here
            5: ("features", self.extract_features),
            6: ("select", self._feature_selection_wrapper),
            7: ("train", self._train_wrapper),
            8: ("save", self.save),
            9: ("test", self.test_new_dataset)
        }

    # ---------------------------
    # VISUALIZATION WRAPPER
    # ---------------------------
    def visualize_current_dataset(self):
        print(">>> Visualizing dataset with all preprocessing steps...")
        if not hasattr(self, "dataset"):
            print("[!] dataset missing → generating labels")
            self.generate_labels()
        visualize_dataset(self.raw_data, self.dataset)

    # wrappers for functions with return values
    def _feature_selection_wrapper(self):
        self.X, self.y = self.feature_selection()

    def _simulation(self):
        simulation_results = run_backtesting_simulator(df=self.dataset)
        print(simulation_results)

    def _train_wrapper(self):
        if not hasattr(self, "X") or not hasattr(self, "y"):
            raise RuntimeError(
                "Features are not generated yet. Run step 4 (select) before step 5 (train)."
            )
        self.train_model(self.X, self.y)

    # ---------------------------------------------------
    # Flexible pipeline with start/end control
    # ---------------------------------------------------
    def run_pipeline(self, start_step_=1, end_step_=9):
        print(f"\n>>> Running pipeline from step {start_step_} to {end_step_}\n")

        for step in range(start_step_, end_step_ + 1):
            if not hasattr(self, "raw_data"):
                print("Auto-running Step 1: LOAD (required for LABEL)")
                self.load_and_prepare_raw_data()

            if not hasattr(self, "dataset"):
                print("Auto-running Step 2: LABEL (required for FEATURES)")
                self.generate_labels()

            # dependency for STEP 4 (feature selection)
            if step == 4:
                if not hasattr(self, "df_features"):
                    print("Auto-running Step 3: FEATURES (required for SELECT)")
                    self.extract_features()

            # dependency for STEP 5 (training)
            if step == 5:
                if not hasattr(self, "X"):
                    print("Auto-running Step 4: SELECT (required for TRAIN)")
                    self._feature_selection_wrapper()

            step_name, func = self.step_functions[step]
            print(f"=== Step {step}: {step_name.upper()} ===")
            func()

    # ---------------------------
    # LOAD + CLEAN DATA
    # ---------------------------
    def load_and_prepare_raw_data(self):
        print("Loading dataset...")
        dt = load_dataset(self.data_dir, self.file_name)
        dt = rename_col(dt)
        self.raw_data = dt
        print(self.raw_data.head(), self.raw_data['Signal'].value_counts())

    # ---------------------------
    # SIGNAL GENERATION
    # ---------------------------
    def generate_labels(self):
        print("Generating labels from raw data...")

        self.dataset = prepare_signal(self.raw_data)
        save_path = os.path.join(self.data_dir, 'cleaned_generated_signal.csv')

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.dataset.to_csv(save_path, index=False)
        print(f"Dataset with generated Signal saved at {save_path}")

    # ---------------------------
    # FEATURE EXTRACTION + CLEANING
    # ---------------------------
    def extract_features(self):
        print("Extracting features...")
        # df_feat = extract_all_features(self.dataset)
        df_feat = extract_fast_features(self.dataset)
        df_feat = handling_nan_after_feature_generate(df_feat)
        self.df_features = df_feat

    # ---------------------------
    # FEATURE SELECTION
    # ---------------------------
    def feature_selection(self):
        print("Performing feature selection...")

        df = self.df_features.dropna(subset=["Signal"]).copy()
        X = df.drop(columns=["Signal"])
        y = df["Signal"]
        X = X.fillna(X.mean())

        selected_features, votes, masks, pipe = select_best_features(X, y, self.n_features)
        selected_features = selected_features[selected_features != "time"]

        self.selected_features = list(selected_features)
        self.pipe = pipe

        # Save internally
        self.X = X
        self.y = y

        pd.Series(self.selected_features).to_csv("selected_features.csv", index=False)
        print("Selected Features:", self.selected_features)

        return X, y

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    def train_model(self, X, y):
        print("Preparing dataset for model training...")
        x_selected = X[self.selected_features]
        x_processed, y_mapped, pipe = prepare_dataset_for_model(x_selected, y)

        print('Training raw XGB model')
        model = xgbmodel(x_processed, y_mapped)

        # print("Training XGBoost (ADASYN)...")
        # model = xgbmodel_adasyn(x_processed, y_mapped)
        #
        # print("Running K-Fold evaluation...")
        # xgbmodel_kfold(model, x_processed, y_mapped)
        #
        # print("Comparing ADASYN & SMOTE performance...")
        # xgbmodel_comparison_with_adasyn_smote(x_processed, y_mapped)

        self.model = model
        self.pipe = pipe

    # ---------------------------
    # SAVE + LOAD MODEL
    # ---------------------------
    def save(self):
        save_model(self.pipe, self.selected_features, self.model)

    def load(self):
        self.pipe, self.selected_features, self.model = load_model()
        print("Model loaded successfully.")

    # ---------------------------
    # TESTING ON NEW DATA
    # ---------------------------
    def test_new_dataset(self):
        print("Loading external test dataset...")
        test_df = load_test_dataset(self.test_file_path)
        test_df = rename_col(test_df)

        print("Extracting features from test dataset...")
        # test_df_features = extract_all_features(test_df.iloc[-10000:, :])
        test_df_features = extract_fast_features(test_df.iloc[-10000:, :])
        test_df_features = handling_nan_after_feature_generate(test_df_features)

        if self.selected_features:
            x = test_df_features[self.selected_features].copy()
        else:
            pipe_, selected_features_, model_ = load_model()
            self.selected_features = selected_features_
            x = test_df_features[self.selected_features].copy()
            self.pipe = pipe_
            self.model = model_

        print("Predicting signals...")
        result_df = predict_with_new_dataset(
            x, self.pipe, self.model,
            test_df_features[self.selected_features]
        )
        # Add 'close' column as an additional feature for plotting result
        result_df['close'] = test_df_features['close']
        result_df.reset_index(drop=True, inplace=True)
        generate_signal_plot(result_df, val_limit=10000)

        return result_df

    # ---------------------------
    # RUN FULL PIPELINE
    # ---------------------------
    def run_full_pipeline(self):
        self.load_and_prepare_raw_data()
        self.generate_labels()
        self.extract_features()
        X, y = self.feature_selection()
        self.train_model(X, y)
        self.save()
        self.test_new_dataset()

        print("\nPipeline completed successfully.")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal ML Pipeline")

    parser.add_argument("--start", type=str, default="1",
                        help="Start step (number or name)")
    parser.add_argument("--end", type=str, default="7",
                        help="End step (number or name)")

    args = parser.parse_args()

    step_map = {
        "load": 1,
        "label": 2,
        "visualize": 3,
        "simulation": 4,
        "features": 5,
        "select": 6,
        "train": 7,
        "save": 8,
        "test": 9
    }


    def convert_step(x):
        if x.isdigit():
            return int(x)
        return step_map[x.lower()]


    start_step = convert_step(args.start)
    end_step = convert_step(args.end)

    # Configure paths
    PROJECT_ROOT = find_project_root()
    DATASETS_DIR = PROJECT_ROOT / "datasets"

    if os.path.exists(DATASETS_DIR):
        data_dir = DATASETS_DIR
    else:
        data_dir = r"D:\Repos_git\Make_Money_with_Tensorflow_2.0\projects\datasets"

    training_file = "Cleaned_Signal_EURUSD_for_training_635_635_60000.csv"
    test_file = os.path.join(data_dir, 'GBPUSD_H1_20140525_20251021.csv')

    # Create pipeline instance
    pipeline = SignalMLPipeline(
        data_dir_=data_dir,
        file_name_=training_file,
        test_file_path_=test_file,
        n_features=20
    )

    # Run with step control
    pipeline.run_pipeline(start_step, end_step)
