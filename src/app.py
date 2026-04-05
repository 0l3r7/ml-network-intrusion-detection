import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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
    run_sniffer_in_thread(interface=None)  # Auto-detect interface
    st.session_state["sniffer_started"] = True

# -----------------------------------
# LOAD MODEL FILES
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model = joblib.load(os.path.join(BASE_DIR, "..", "models", "rf_model.pkl"))
iso_model = joblib.load(os.path.join(BASE_DIR, "..", "models", "iso_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "..", "models", "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "..", "models", "columns.pkl"))

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🛡️ Machine Learning-Based Network Intrusion Detection System")
st.caption("Real-Time Nmap Detection + CSV-Based ML Intrusion Detection")

# -----------------------------------
# LIVE ALERT SECTION
# -----------------------------------
st.subheader("🚨 Live Nmap Scan Detection")

if live_alerts:
    live_df = pd.DataFrame(live_alerts)

    st.error("⚠ Live suspicious scan activity detected!")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Alerts", len(live_df))
    col2.metric("Unique Attackers", live_df["src_ip"].nunique())
    col3.metric("Last Alert", live_df.iloc[-1]["timestamp"])

    st.dataframe(live_df, use_container_width=True)

    # Live alert bar chart
    attacker_counts = live_df["src_ip"].value_counts()
    fig_live, ax_live = plt.subplots()
    attacker_counts.plot(kind="bar", ax=ax_live)
    ax_live.set_title("Live Alerts by Source IP")
    ax_live.set_xlabel("Source IP")
    ax_live.set_ylabel("Alert Count")
    st.pyplot(fig_live)

else:
    st.info("No live scan alerts yet. Run Nmap from the attacker VM, then click Rerun / refresh.")

st.divider()

# -----------------------------------
# CSV ML SECTION
# -----------------------------------
st.subheader("📂 CSV-Based ML Intrusion Detection")
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, header=None)
        label_column = data.columns[-2]

        # Build features
        X = pd.get_dummies(data.drop(label_column, axis=1))
        X = X.reindex(columns=columns, fill_value=0)
        X_scaled = scaler.transform(X)

        # Predict
        predictions = rf_model.predict(X_scaled)
        data["prediction"] = predictions

        # Create readable attack_type column
        data["attack_type"] = data["prediction"].apply(lambda x: "Attack" if x == 1 else "Normal")

        attacks = data[data["prediction"] == 1]
        normal = data[data["prediction"] == 0]

        # Metrics
        st.subheader("📊 Traffic Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Traffic", len(data))
        c2.metric("Detected Attacks", len(attacks))
        c3.metric("Normal Traffic", len(normal))

        if len(attacks) > 0:
            st.error("⚠ Intrusion Detected!")
        else:
            st.success("✅ No attacks detected in uploaded file.")

        # Show data
        st.subheader("📋 Prediction Results")
        st.dataframe(data, use_container_width=True)

        # -----------------------------------
        # CHARTS
        # -----------------------------------
        st.subheader("📈 Visual Analytics")

        chart_col1, chart_col2 = st.columns(2)

        # Pie chart
        with chart_col1:
            traffic_counts = data["prediction"].value_counts()
            normal_count = traffic_counts.get(0, 0)
            attack_count = traffic_counts.get(1, 0)

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
        with chart_col2:
            attack_types = data[data["prediction"] == 1]["attack_type"].value_counts()
            if not attack_types.empty:
                fig2, ax2 = plt.subplots()
                attack_types.plot(kind="bar", ax=ax2)
                ax2.set_title("Attack Types Detected")
                ax2.set_xlabel("Attack Type")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
            else:
                st.info("No attack types to display.")

        # -----------------------------------
        # SIMPLE TRAFFIC TREND (line chart)
        # -----------------------------------
        st.subheader("📉 Traffic Trend (by Row Index)")
        trend_df = data["prediction"].reset_index()
        trend_df.columns = ["Traffic Index", "Prediction"]

        st.line_chart(trend_df.set_index("Traffic Index"))

    except Exception as e:
        st.error(f"Error processing CSV: {e}")