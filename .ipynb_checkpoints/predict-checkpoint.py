import pandas as pd
import joblib

# Load files
rf_model = joblib.load("rf_model.pkl")
iso_model = joblib.load("iso_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Load new traffic file (use same format as training dataset)
data = pd.read_csv("dataset/network.csv", header=None)

# Label column
label_column = data.columns[-2]

# Keep original labels for display
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

# Random Forest prediction
rf_predictions = rf_model.predict(X_scaled)

# Isolation Forest prediction
iso_raw = iso_model.predict(X_scaled)
iso_predictions = [0 if p == 1 else 1 for p in iso_raw]

# Add results
results = pd.DataFrame({
    "Actual_Label": original_labels,
    "RF_Prediction": ["BENIGN" if p == 0 else "ATTACK" for p in rf_predictions],
    "ISO_Prediction": ["BENIGN" if p == 0 else "ATTACK" for p in iso_predictions]
})

print(results.head(20))