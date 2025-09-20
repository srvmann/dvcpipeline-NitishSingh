import os
import pandas as pd
import scipy.sparse
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report

print("--- Starting Model Evaluation ---")

# --- 1. Define Paths and Create Directories ---
# Path to the trained model
model_path = os.path.join("models", "model.pkl")

# Path to the test data
data_path = os.path.join("data", "processed")
test_features_path = os.path.join(data_path, "test_vectorized.npz")
test_labels_path = os.path.join("data", "raw", "test.csv")

# Path for metrics output
metrics_dir = "metrics"
os.makedirs(metrics_dir, exist_ok=True)
metrics_path = os.path.join(metrics_dir, "eval_metrics.json")


# --- 2. Load the Model and Test Data ---
print("Loading model and test data...")
try:
    # Load the trained model from the .pkl file
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the test feature matrix
    X_test = scipy.sparse.load_npz(test_features_path)

    # Load the test labels from the original raw CSV
    y_test = pd.read_csv(test_labels_path)['label']
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure the model training and data processing stages have been run.")
    exit()
except KeyError:
    print("Error: Ensure the target column in 'data/raw/test.csv' is named 'label'.")
    exit()


# --- 3. Make Predictions ---
print("Making predictions on the test set...")
y_pred = model.predict(X_test)


# --- 4. Calculate and Display Metrics ---
print("Calculating metrics...")
accuracy = accuracy_score(y_test, y_pred)
# Generate a full classification report (precision, recall, f1-score)
# output_dict=True makes it easy to save as JSON
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
# Pretty print the report for the console
print(classification_report(y_test, y_pred))


# --- 5. Save Metrics to JSON File ---
# We will save the key metrics for DVC to track
metrics_to_save = {
    'accuracy': accuracy,
    'macro_avg_precision': report['macro avg']['precision'],
    'macro_avg_recall': report['macro avg']['recall'],
    'macro_avg_f1-score': report['macro avg']['f1-score'],
    'weighted_avg_f1-score': report['weighted avg']['f1-score']
}

print(f"\nSaving metrics to {metrics_path}...")
with open(metrics_path, 'w') as f:
    json.dump(metrics_to_save, f, indent=4)

print("--- Model Evaluation Finished Successfully ---")
