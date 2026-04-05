import os
import pandas as pd
import joblib

# ----------------------------
# BASE PATH SETUP
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "network.csv")

# ----------------------------
# LOAD MODELS
# ----------------------------
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))

# ----------------------------
# LOAD DATA
# ----------------------------
print("Loading dataset from:", DATA_PATH)
data = pd.read_csv(DATA_PATH, header=None)

# Label column
label_column = data.columns[-2]

# Keep original labels
original_labels = data[label_column].copy()

# Remove label column
X = data.drop(label_column, axis=1)

# One-hot encode
X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

# Match training columns
X = X.reindex(columns=columns, fill_value=0)

# Scale
X_scaled = scaler.transform(X)

# ----------------------------
# PREDICTIONS
# ----------------------------
rf_predictions = rf_model.predict(X_scaled)

iso_raw = iso_model.predict(X_scaled)
iso_predictions = [0 if p == 1 else 1 for p in iso_raw]

# ----------------------------
# RESULTS
# ----------------------------
results = pd.DataFrame({
    "Actual_Label": original_labels,
    "RF_Prediction": ["BENIGN" if p == 0 else "ATTACK" for p in rf_predictions],
    "ISO_Prediction": ["BENIGN" if p == 0 else "ATTACK" for p in iso_predictions]
})

print(results.head(20))