import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="NIDS Dashboard", layout="wide")

# Load model files
rf_model = joblib.load("rf_model.pkl")
iso_model = joblib.load("iso_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Network Intrusion Detection System Dashboard")
st.write("Upload a network traffic CSV file to detect possible attacks.")

uploaded_file = st.file_uploader("Upload Network Traffic CSV")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file, header=None)
    label_column = data.columns[-2]

    # Save original labels
    original_labels = data[label_column].copy()

    # Preprocess
    X = pd.get_dummies(data.drop(label_column, axis=1))
    X.columns = X.columns.astype(str)
    X = X.reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X)

    # Random Forest predictions
    rf_predictions = rf_model.predict(X_scaled)

    # Isolation Forest predictions
    iso_raw = iso_model.predict(X_scaled)
    iso_predictions = [0 if p == 1 else 1 for p in iso_raw]

    # Choose main displayed prediction = Random Forest
    data["prediction"] = rf_predictions

    # Attack type column (for display)
    data["attack_type"] = original_labels.apply(lambda x: "Normal" if x == "normal" else str(x))

    # Summary counts
    attacks = data[data["prediction"] == 1]
    normal = data[data["prediction"] == 0]

    # Live counters
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Traffic", len(data))
    col2.metric("Detected Attacks", len(attacks))
    col3.metric("Normal Traffic", len(normal))

    # Alert
    if len(attacks) > 0:
        st.error("⚠ Intrusion Detected!")
    else:
        st.success(" No intrusion detected.")

    # Alert table
    st.subheader(" Alert Table (Detected Attacks)")
    if len(attacks) > 0:
        st.dataframe(attacks.head(50))
    else:
        st.write("No attacks detected in this file.")

    # Full data
    st.subheader(" Full Traffic Data with Predictions")
    st.dataframe(data)

    # Pie chart: normal vs attack
    st.subheader(" Traffic Distribution")
    traffic_counts = data["prediction"].value_counts().reindex([0, 1], fill_value=0)

    fig1, ax1 = plt.subplots()
    ax1.pie(
        traffic_counts,
        labels=["Normal", "Attack"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#F44336"]
    )
    ax1.axis("equal")
    st.pyplot(fig1)

    # Pie chart of attack types (actual labels among predicted attacks)
    st.subheader(" Attack Types Detected")
    attack_types = data[data["prediction"] == 1]["attack_type"].value_counts()

    if len(attack_types) > 0:
        fig2, ax2 = plt.subplots()
        ax2.pie(
            attack_types,
            labels=attack_types.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)
    else:
        st.write("No attack types to display.")

    # Line chart: traffic over time (connection index)
    st.subheader(" Traffic Over Time")
    line_data = pd.DataFrame({
        "Connection Index": range(len(data)),
        "Attack Flag": [1 if p == 1 else 0 for p in data["prediction"]]
    }).set_index("Connection Index")

    st.line_chart(line_data)

    # Optional: compare RF vs Isolation Forest
    st.subheader(" Model Comparison Preview")
    compare_df = pd.DataFrame({
        "Actual Label": original_labels.head(20).values,
        "Random Forest": ["BENIGN" if p == 0 else "ATTACK" for p in rf_predictions[:20]],
        "Isolation Forest": ["BENIGN" if p == 0 else "ATTACK" for p in iso_predictions[:20]]
    })

    st.dataframe(compare_df)