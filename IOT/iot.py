import streamlit as st
from streamlit_autorefresh import st_autorefresh
import json
import os
import pandas as pd
import joblib
from datetime import datetime
import plotly.express as px

# ------------------------
# Config
# ------------------------
DATA_FILE = "iot_messages.json"
MODEL_PATH = "maintenance_model.pkl"

st.set_page_config(page_title="IIoT Predictive Dashboard", layout="wide")

# ------------------------
# Load External CSS
# ------------------------
def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("IOT/iot.css")

# ------------------------
# Title
# ------------------------
st.title("📊 Real-Time Monitoring Of Machines with Predictive Maintenance")
st.markdown("<hr>", unsafe_allow_html=True)

# Auto-refresh every 10 seconds
st_autorefresh(interval=10000, key="datareframe")

# ------------------------
# Load ML model
# ------------------------
clf = None
if os.path.exists(MODEL_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        st.success("✅ Predictive Maintenance Model Loaded")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning("❌ Model file not found")

# ------------------------
# Load all messages
# ------------------------
def load_messages():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

messages = load_messages()

if messages:
    df = pd.DataFrame(messages)
    df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.Series([datetime.now()]*len(df))))

    # ------------------------
    # Predictive Maintenance
    # ------------------------
    if clf:
        try:
            df_model = df[["Temperature_C","Vibration_Level_mms","Energy_Consumption_kWh",
                           "Carbon_Emission_kg","Downtime_Hours","Industry","Region"]]
            preds = clf.predict(df_model)
            probs = clf.predict_proba(df_model)[:,1]
            df["maintenance_required"] = preds
            df["risk_probability"] = probs
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # ------------------------
    # Hierarchical Machine Selector (Top)
    # ------------------------
    st.subheader("🔽 Select Machine")
    industries = sorted(df["Industry"].unique())
    selected_industry = st.selectbox("Select Sector (Industry)", industries)

    companies = sorted(df[df["Industry"] == selected_industry]["Company_ID"].unique())
    selected_company = st.selectbox("Select Company", companies)

    machines = sorted(df[(df["Industry"] == selected_industry) & (df["Company_ID"] == selected_company)]["Machine_ID"].unique())
    selected_machine = st.selectbox("Select Machine", machines)

    # Filter data for selected machine
    machine_df = df[df["Machine_ID"] == selected_machine]
    latest_data = machine_df.iloc[-1]

    # ------------------------
    # Machine Info Card
    # ------------------------
    st.subheader(f"🏭 Machine Information - {selected_machine}")
    col1, col2, col3, col4 = st.columns(4)
    col1.info(f"Company ID: {latest_data.get('Company_ID')}\nIndustry: {latest_data.get('Industry')}")
    col2.info(f"Region: {latest_data.get('Region')}")
    col3.info(f"Machine ID: {latest_data.get('Machine_ID')}")
    col4.info(f"Safety Incidents: {latest_data.get('Safety_Incidents')}")

    # ------------------------
    # Sensor Metrics
    # ------------------------
    st.subheader("📊 Latest Sensor Readings")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡 Temperature (°C)", latest_data.get("Temperature_C"))
    col1.metric("📈 Vibration (mm/s)", latest_data.get("Vibration_Level_mms"))
    col2.metric("⚡ Energy (kWh)", latest_data.get("Energy_Consumption_kWh"))
    col2.metric("🌍 Carbon Emission (kg)", latest_data.get("Carbon_Emission_kg"))
    col3.metric("⏱ Downtime (hours)", latest_data.get("Downtime_Hours"))
    col3.metric("🚨 Safety Incidents", latest_data.get("Safety_Incidents"))

    # ------------------------
    # Maintenance Status Card with Alerts
    # ------------------------
    if "maintenance_required" in machine_df.columns:
        status = "⚠️ Maintenance Required" if latest_data.get("maintenance_required") else "✅ No Maintenance Needed"
        color = "red" if latest_data.get("maintenance_required") else "green"
        risk_prob = latest_data.get("risk_probability", 0)
        risk_pct = f"{float(risk_prob)*100:.1f}%" if risk_prob is not None else "N/A"
        st.subheader("🛠 Predictive Maintenance")
        st.markdown(
            f"<h3 style='color:{color}'>{status} &nbsp; <span style='font-size:1.1em;'>({risk_pct} Risk)</span></h3>",
            unsafe_allow_html=True
        )
        
        if risk_prob is not None:
            try:
                risk_prob_float = float(risk_prob)
                st.progress(risk_prob_float)
                if risk_prob_float > 0.8:
                    st.error("🚨 HIGH RISK: Immediate maintenance recommended!")
                elif risk_prob_float > 0.5:
                    st.warning("⚠️ Moderate Risk: Monitor closely.")
                else:
                    st.success("✅ Low Risk")
            except Exception as e:
                st.error(f"❌ Failed to display progress: {e}")

    # ------------------------
    # Maintenance Percentage Summary
    # ------------------------
    if "maintenance_required" in df.columns:
        total_machines = len(df)
        high_maint = df["maintenance_required"].sum()
        low_maint = total_machines - high_maint
        high_pct = (high_maint / total_machines) * 100 if total_machines else 0
        low_pct = (low_maint / total_machines) * 100 if total_machines else 0

        colA, colB = st.columns(2)
        colA.metric("⚠️ High Maintenance Needed (%)", f"{high_pct:.1f}%", f"{int(high_maint)} machines")
        colB.metric("✅ Low Maintenance Needed (%)", f"{low_pct:.1f}%", f"{int(low_maint)} machines")
    # ------------------------
    # Date Range Filter
    # ------------------------
    st.subheader("📅 Filter Historical Data")
    start_date, end_date = st.date_input(
        "Select Date Range", 
        [df["timestamp"].min().date(), df["timestamp"].max().date()]
    )
    # Filter ALL machines for the selected date range
    filtered_df = df[
        (df["timestamp"] >= pd.to_datetime(start_date)) & 
        (df["timestamp"] <= pd.to_datetime(end_date))
    ]
    # Add serial number column
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df.insert(0, "Serial", filtered_df.index + 1)
    st.dataframe(filtered_df.tail(400))

    # ------------------------
    # Risk Machines Overview (Filtered)
    # ------------------------
    st.subheader("🚨 Top 5 High-Risk Machines")
    if not filtered_df.empty and "risk_probability" in filtered_df.columns:
        # Remove rows with missing Machine_ID or risk_probability
        risk_df = filtered_df.dropna(subset=["Machine_ID", "risk_probability"])
        if not risk_df.empty:
            top_risk = (
                risk_df.groupby("Machine_ID")["risk_probability"]
                .max()  # Use max instead of mean
                .nlargest(5)
                .reset_index()
            )
            st.table(top_risk)
        else:
            st.info("No valid risk data available for the selected date range.")
    else:
        st.info("No data available for the selected date range.")

    # ------------------------
    # Machine Comparison
    # ------------------------
    st.subheader("📊 Compare Machines")
    selected_machines = st.multiselect(
        "Select Machines to Compare", 
        sorted(df["Machine_ID"].unique()), 
        default=[selected_machine]
    )
    compare_df = df[df["Machine_ID"].isin(selected_machines)]
    if not compare_df.empty:
        fig_compare = px.line(compare_df, x="timestamp", y="Temperature_C", color="Machine_ID", 
                              title="🌡 Temperature Comparison")
        st.plotly_chart(fig_compare, use_container_width=True)

    # ------------------------
    # Sector-Level Summary
    # ------------------------
    st.subheader("📈 Sector-Level Summary")
    sector_summary = df.groupby("Industry").agg({
        "Temperature_C": "mean",
        "Vibration_Level_mms": "mean",
        "Energy_Consumption_kWh": "mean",
        "Carbon_Emission_kg": "mean",
        "Downtime_Hours": "mean",
        "maintenance_required": "sum"
    }).reset_index()


    # Pie chart for Total Machines at Risk per Sector
    fig_sector_pie = px.pie(
        sector_summary,
        names="Industry",
        values="maintenance_required",
        title="📊 Total Machines at Risk per Sector",
        color="Industry"
    )
    fig_sector_pie.update_traces(textinfo="label+percent+value")
    st.plotly_chart(fig_sector_pie, use_container_width=True)


    # ------------------------
    # What-If Simulator
    # ------------------------
    st.sidebar.header("🔧 What-If Simulator")
    temp = st.sidebar.slider("Temperature (°C)", 20, 120, 60)
    vib = st.sidebar.slider("Vibration (mm/s)", 0, 50, 10)
    energy = st.sidebar.slider("Energy Consumption (kWh)", 10, 1800, 100)
    carbon = st.sidebar.slider("Carbon Emission (kg)", 1, 700, 10)
    downtime = st.sidebar.slider("Downtime (hours)", 0, 24, 1)

    if clf:
        sim_df = pd.DataFrame([[temp, vib, energy, carbon, downtime, selected_industry, "RegionX"]],
                             columns=["Temperature_C","Vibration_Level_mms","Energy_Consumption_kWh",
                                      "Carbon_Emission_kg","Downtime_Hours","Industry","Region"])
        sim_pred = clf.predict(sim_df)[0]
        sim_prob = clf.predict_proba(sim_df)[0][1]
        st.sidebar.markdown(f"### Prediction: {'⚠️ Maintenance Needed' if sim_pred else '✅ Safe'}")
        st.sidebar.progress(float(sim_prob))
        st.sidebar.write(f"Risk Probability: {sim_prob*100:.2f}%")

    # ------------------------
    # Download Option
    # ------------------------
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

else:
    st.warning("⏳ Waiting for MQTT messages...")
