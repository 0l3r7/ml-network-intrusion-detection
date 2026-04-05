import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ----------------------------
# BASE PATH SETUP (IMPORTANT)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path (inside src/dataset/)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "network.csv")

# Models path (outside src -> /models/)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
print("Loading dataset from:", DATA_PATH)
data = pd.read_csv(DATA_PATH, header=None)

label_column = data.columns[-2]
data[label_column] = data[label_column].apply(lambda x: 0 if x == "normal" else 1)

# ----------------------------
# Features & target
# ----------------------------
X = pd.get_dummies(data.drop(label_column, axis=1))
X.columns = X.columns.astype(str)
y = data[label_column]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Random Forest
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate RF
pred = rf_model.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# ----------------------------
# Train Isolation Forest
# ----------------------------
X_benign = X[y == 0]

iso_scaler = StandardScaler()
X_benign_scaled = iso_scaler.fit_transform(X_benign)

iso_model = IsolationForest(
    n_estimators=500,
    contamination=0.01,
    random_state=42
)
iso_model.fit(X_benign_scaled)

# ----------------------------
# Save everything to /models/
# ----------------------------
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(X.columns, os.path.join(MODEL_DIR, "columns.pkl"))

joblib.dump(iso_model, os.path.join(MODEL_DIR, "iso_model.pkl"))
joblib.dump(iso_scaler, os.path.join(MODEL_DIR, "iso_scaler.pkl"))

print("\n✅ All models saved to:", MODEL_DIR)