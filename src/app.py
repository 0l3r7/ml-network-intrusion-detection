import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.live_nmap_detector import run_sniffer_in_thread, live_alerts
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="ML-Based NIDS with Real-Time Alerting",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------
# START LIVE SNIFFER ONCE
# -----------------------------------
if "sniffer_started" not in st.session_state:
    try:
        run_sniffer_in_thread(interface=None)
        st.session_state["sniffer_started"] = True
    except Exception:
        st.warning("Live packet sniffing not available (Npcap/WinPcap not installed).")

# -----------------------------------
# LOAD MODEL FILES
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

rf_model = joblib.load(os.path.join(MODEL_PATH, "best_model.pkl"))
iso_model = joblib.load(os.path.join(MODEL_PATH, "iso_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_PATH, "columns.pkl"))

# -----------------------------------
# TITLE
# -----------------------------------
st.title("Machine Learning-Based Network Intrusion Detection System")
st.caption("Real-Time Nmap Detection + CSV-Based ML Intrusion Detection")

# -----------------------------------
# LIVE ALERT SECTION
# -----------------------------------
st.subheader("Live Nmap Scan Detection")

if live_alerts:
    live_df = pd.DataFrame(live_alerts)

    st.error("Live suspicious scan activity detected")

    col1, col2, col3 = st.columns(3)
    col1.metric("Live Alerts", len(live_df))
    col2.metric("Unique Attackers", live_df["src_ip"].nunique())
    col3.metric("Last Alert", live_df.iloc[-1]["timestamp"])

    st.dataframe(live_df, use_container_width=True)

    attacker_counts = live_df["src_ip"].value_counts()
    fig_live, ax_live = plt.subplots()
    attacker_counts.plot(kind="bar", ax=ax_live)
    ax_live.set_title("Live Alerts by Source IP")
    ax_live.set_xlabel("Source IP")
    ax_live.set_ylabel("Alert Count")
    st.pyplot(fig_live)

else:
    st.info("No live scan alerts yet. Run Nmap from attacker machine and refresh.")

st.divider()

# -----------------------------------
# CSV ML SECTION
# -----------------------------------
st.subheader("CSV-Based ML Intrusion Detection")
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, header=None)
        label_column = data.columns[-2]

        # Feature processing
        X = pd.get_dummies(data.drop(label_column, axis=1))
        X = X.reindex(columns=columns, fill_value=0)
        X_scaled = scaler.transform(X)

        # Prediction
        predictions = rf_model.predict(X_scaled)
        data["prediction"] = predictions
        data["attack_type"] = data["prediction"].apply(
            lambda x: "Attack" if x == 1 else "Normal"
        )

        attacks = data[data["prediction"] == 1]
        normal = data[data["prediction"] == 0]

        # Metrics
        st.subheader("Traffic Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Traffic", len(data))
        c2.metric("Detected Attacks", len(attacks))
        c3.metric("Normal Traffic", len(normal))

        if len(attacks) > 0:
            st.error("Intrusion Detected")
        else:
            st.success("No attacks detected")

        # Table
        st.subheader("Prediction Results")
        st.dataframe(data, use_container_width=True)

        # -----------------------------------
        # CONFUSION MATRIX
        # -----------------------------------
        try:
            y_true = data[label_column].apply(
                lambda x: 0 if str(x).lower() == "normal" else 1
            )
            y_pred = data["prediction"]

            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()

            st.subheader("Confusion Matrix Analysis")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("True Negative", TN)
            m2.metric("False Positive", FP)
            m3.metric("False Negative", FN)
            m4.metric("True Positive", TP)

            st.info(
                f"False Positives: {FP} normal traffic flagged as attack | "
                f"False Negatives: {FN} attacks missed (critical)"
            )

            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm)

            ax_cm.set_title("Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")

            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Normal", "Attack"])
            ax_cm.set_yticklabels(["Normal", "Attack"])

            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, cm[i][j], ha="center", va="center")

            st.pyplot(fig_cm)

        except Exception:
            st.warning("Confusion matrix not available (no labels detected).")

        # -----------------------------------
        # CHARTS
        # -----------------------------------
        st.subheader("Visual Analytics")

        colA, colB = st.columns(2)

        # Pie chart
        with colA:
            counts = data["prediction"].value_counts()
            normal_count = counts.get(0, 0)
            attack_count = counts.get(1, 0)

            fig1, ax1 = plt.subplots()
            ax1.pie(
                [normal_count, attack_count],
                labels=["Normal", "Attack"],
                autopct="%1.1f%%",
                startangle=90
            )
            ax1.axis("equal")
            ax1.set_title("Traffic Distribution")
            st.pyplot(fig1)

        # Bar chart
        with colB:
            attack_types = data[data["prediction"] == 1]["attack_type"].value_counts()
            if not attack_types.empty:
                fig2, ax2 = plt.subplots()
                attack_types.plot(kind="bar", ax=ax2)
                ax2.set_title("Attack Types")
                ax2.set_xlabel("Type")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
            else:
                st.info("No attack types detected.")

        # Line chart
        st.subheader("Traffic Trend")
        trend_df = data["prediction"].reset_index()
        trend_df.columns = ["Index", "Prediction"]

        st.line_chart(trend_df.set_index("Index"))

    except Exception as e:
        st.error(f"Error processing CSV: {e}")