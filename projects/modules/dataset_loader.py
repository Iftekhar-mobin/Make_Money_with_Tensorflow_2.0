import os
import pandas as pd


def load_dataset(data_directory, file_name):
    # 1. Full path inside provided directory
    file_loc = os.path.join(data_directory, file_name)

    if os.path.exists(file_loc):
        return pd.read_csv(file_loc, index_col=0)

    # 2. Try loading from ./datasets/ (current directory fallback)
    fallback_dir = os.path.join(os.curdir, "datasets")
    fallback_path = os.path.join(fallback_dir, file_name)

    if os.path.exists(fallback_path):
        print(f"Warning: File not found in '{data_directory}', loaded from fallback '{fallback_dir}'")
        return pd.read_csv(fallback_path, index_col=0)

    # 3. File not found anywhere → raise error
    raise FileNotFoundError(
        f"\nDataset file '{file_name}' not found.\n"
        f"Searched locations:\n"
        f"  → {file_loc}\n"
        f"  → {fallback_path}\n\n"
        f"Please download the dataset and place it inside:\n"
        f"  ✔ {data_directory}\n"
        f"or\n"
        f"  ✔ {fallback_dir}\n"
    )


def load_test_dataset(test_file_path):
    test_df = pd.read_csv(test_file_path, index_col=0)
    test_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
                   inplace=True)
    test_df.drop(columns=['Time'], errors='ignore', inplace=True)
    test_df.reset_index(drop=True)
    return test_df
