import os
import yaml
import pandas as pd
import scipy.sparse
import pickle
from sklearn.linear_model import LogisticRegression

# Importing the parameter from params.yaml file
params =  yaml.safe_load(open("params.yaml","r"))["modelbuilding"]

print("--- Starting Model Building Stage ---")

# --- 1. Define Paths and Create Output Directory ---
# Path to processed data
data_path = os.path.join("data", "processed")

# Path for model output - THIS IS THE CRUCIAL FIX
model_dir = "models"
os.makedirs(model_dir, exist_ok=True) # Create the directory if it doesn't exist
model_path = os.path.join(model_dir, "model.pkl")

print(f"Model will be saved to: {model_path}")


# --- 2. Load the Data ---
print("Loading features and labels...")
try:
    X_train = scipy.sparse.load_npz(os.path.join(data_path, "train_vectorized.npz"))
    X_test = scipy.sparse.load_npz(os.path.join(data_path, "test_vectorized.npz"))
    y_train = pd.read_csv("./data/raw/train.csv")['label']
    y_test = pd.read_csv("./data/raw/test.csv")['label']
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the data preprocessing stage has been run successfully.")
    exit()
except KeyError:
    print("Error: Please make sure your target column is named 'label' in your raw CSVs.")
    exit()


# --- 3. Build and Train the Model ---
print("Training the Logistic Regression model...")
model = LogisticRegression(random_state = params["random_state"],max_iter = params["max_iter"])
model.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Save the Model ---
print(f"Saving model to disk at {model_path}...")
try:
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

print("--- Model Building Stage Finished ---")
