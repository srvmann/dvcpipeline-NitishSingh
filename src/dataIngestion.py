import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Load test_size parameter safely
# ----------------------------
def load_params(params_path):
    """
    Load test_size parameter from params.yaml.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        test_size = params.get("dataingestion", {}).get("test_size", None)
        if test_size is None:
            raise ValueError("'test_size' not found in params.yaml under 'dataingestion'")
        print(f"Loaded test_size = {test_size}")
        return test_size
    except FileNotFoundError:
        print(f"Error: params.yaml file not found at {params_path}")
        return None
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while loading parameters: {e}")
        return None


# ----------------------------
# Load and combine datasets
# ----------------------------
def load_and_combine_news_datasets(fake_news_path, true_news_path):
    """
    Loads fake and true news datasets, assigns labels, and combines them.
    """
    try:
        if not os.path.exists(fake_news_path):
            raise FileNotFoundError(f"Fake news file not found: {fake_news_path}")
        if not os.path.exists(true_news_path):
            raise FileNotFoundError(f"True news file not found: {true_news_path}")

        fake_news_df = pd.read_csv(fake_news_path)
        fake_news_df["label"] = "fake"
        print(f"Loaded fake news: {fake_news_path} | Shape: {fake_news_df.shape}")

        true_news_df = pd.read_csv(true_news_path)
        true_news_df["label"] = "true"
        print(f"Loaded true news: {true_news_path} | Shape: {true_news_df.shape}")

        combined_df = pd.concat([true_news_df, fake_news_df], ignore_index=True)
        print(f"Combined dataset created | Shape: {combined_df.shape}")
        return combined_df

    except pd.errors.EmptyDataError:
        print("Error: One or both CSV files are empty.")
        return None
    except Exception as e:
        print(f"Unexpected error while loading datasets: {e}")
        return None


# ----------------------------
# Preprocess dataset
# ----------------------------
def preprocess_news_data(df):
    """
    Preprocesses the combined news dataset.
    """
    if df is None:
        print("Error: Input DataFrame is None.")
        return None

    try:
        df_processed = df.copy()

        # Convert date column to datetime
        if "date" in df_processed.columns:
            df_processed["date"] = pd.to_datetime(df_processed["date"], errors='coerce', dayfirst = True,format = "mixed")
            dropped_rows = df_processed["date"].isna().sum()
            df_processed.dropna(subset=["date"], inplace=True)
            print(f"Dropped {dropped_rows} rows due to invalid dates.")
        else:
            print("Warning: 'date' column missing. Skipping date processing.")

        # Replace labels
        if "label" in df_processed.columns:
            df_processed["label"].replace({"fake": 0, "true": 1}, inplace=True)
        else:
            print("Warning: 'label' column missing.")

        print(f"Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


# ----------------------------
# Save train/test splits safely
# ----------------------------
def save_data(data_path, train_data, test_data):
    """
    Save train and test data to CSV files.
    """
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        print(f"Data saved in {data_path}")
    except PermissionError:
        print(f"Permission denied: Could not write to {data_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


# ----------------------------
# Main workflow
# ----------------------------
def main():
    params_path = "params.yaml"
    test_size = load_params(params_path)

    if test_size is None:
        print("Aborting: test_size parameter could not be loaded.")
        return

    df = load_and_combine_news_datasets(
        fake_news_path = r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Datasets\Fake-Real-News\Fake.csv",
        true_news_path = r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Datasets\Fake-Real-News\True.csv"
    )

    final_df = preprocess_news_data(df)
    if final_df is None:
        print("Aborting: Preprocessing failed.")
        return

    try:
        train_data, test_data = train_test_split(final_df, test_size = test_size, stratify = final_df["label"], random_state = 42)
    except Exception as e:
        print(f"Error during train/test split: {e}")
        return

    data_path = os.path.join("data", "raw")
    save_data(data_path, train_data, test_data)


if __name__ == "__main__":
    main()
