
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor

st.set_page_config(
    page_title="EnergyTwin AI",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EnergyTwin AI")
st.caption("AI-Powered Building Energy Optimization & Digital Twin")

# Sidebar settings
st.sidebar.header("Project Settings")
building = st.sidebar.selectbox(
    "Select Building",
    ["Engineering College", "Office Building", "Smart Home"]
)

rate = st.sidebar.number_input(
    "Electricity price (₹ per kWh)",
    min_value=1.0,
    max_value=30.0,
    value=8.0,
    step=0.5
)

# Generate demonstration data
@st.cache_data
def generate_data():
    np.random.seed(42)
    days = 120
    rooms = ["Classroom", "Computer Lab", "Library", "Office", "Cafeteria"]
    records = []

    for day in range(days):
        date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days-day-1)

        for room in rooms:
            base = {
                "Classroom": 34,
                "Computer Lab": 68,
                "Library": 25,
                "Office": 30,
                "Cafeteria": 42
            }[room]

            usage = max(5, np.random.normal(base, base * 0.12))

            # Simulated unusual readings
            if np.random.random() < 0.035:
                usage *= np.random.uniform(1.7, 2.2)

            records.append({
                "Date": date,
                "Room": room,
                "Energy_kWh": round(usage, 2)
            })

    return pd.DataFrame(records)

df = generate_data()

# AI anomaly detection
detector = IsolationForest(
    contamination=0.035,
    random_state=42
)
df["Anomaly"] = detector.fit_predict(df[["Energy_kWh"]])
df["Status"] = np.where(
    df["Anomaly"] == -1,
    "Unusual Usage",
    "Normal"
)
df["Cost_Rs"] = df["Energy_kWh"] * rate

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "AI Waste Detection",
        "Energy Prediction",
        "Digital Twin Simulator",
        "Sustainability Report"
    ]
)

# Dashboard
if page == "Dashboard":
    st.subheader(f"{building} — Energy Overview")

    total_energy = df["Energy_kWh"].sum()
    total_cost = total_energy * rate
    avg_daily = df.groupby("Date")["Energy_kWh"].sum().mean()
    unusual = int((df["Status"] == "Unusual Usage").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Energy", f"{total_energy:,.0f} kWh")
    c2.metric("Estimated Cost", f"₹{total_cost:,.0f}")
    c3.metric("Average Daily Use", f"{avg_daily:,.1f} kWh")
    c4.metric("Unusual Readings", unusual)

    st.markdown("### Daily Energy Consumption")
    daily = df.groupby("Date", as_index=False)["Energy_kWh"].sum()
    fig = px.line(
        daily,
        x="Date",
        y="Energy_kWh",
        markers=True,
        title="Building Energy Trend"
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Room-wise Consumption")
    room_data = df.groupby("Room", as_index=False)["Energy_kWh"].sum()
    fig2 = px.bar(
        room_data,
        x="Room",
        y="Energy_kWh",
        color="Room",
        title="Total Energy by Room"
    )
    st.plotly_chart(fig2, width="stretch")

    st.info(
        "Demo mode: readings are simulated for demonstration, "
        "not collected from real electrical meters."
    )

# AI Waste Detection
elif page == "AI Waste Detection":
    st.subheader("🤖 AI Energy Waste Detection")
    st.write(
        "Isolation Forest identifies readings that differ "
        "significantly from typical energy usage."
    )

    anomalies = df[df["Status"] == "Unusual Usage"].copy()
    st.metric("Flagged Readings", len(anomalies))

    fig = px.scatter(
        df,
        x="Date",
        y="Energy_kWh",
        color="Status",
        hover_data=["Room"],
        title="Normal vs Unusual Energy Readings"
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        anomalies[["Date", "Room", "Energy_kWh", "Status"]],
        width="stretch"
    )

    st.caption(
        "An unusual reading is a review alert, not proof of energy waste."
    )

# Energy Prediction
elif page == "Energy Prediction":
    st.subheader("📈 Energy Consumption Prediction")

    daily = df.groupby("Date", as_index=False)["Energy_kWh"].sum()
    daily["Day_Number"] = np.arange(len(daily))

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(daily[["Day_Number"]], daily["Energy_kWh"])

    next_days = st.slider(
        "Days to forecast",
        min_value=7,
        max_value=30,
        value=14
    )

    future_x = np.arange(
        len(daily),
        len(daily) + next_days
    ).reshape(-1, 1)

    predictions = model.predict(future_x)

    future_dates = pd.date_range(
        daily["Date"].max() + pd.Timedelta(days=1),
        periods=next_days
    )

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted_kWh": predictions
    })

    st.metric(
        "Average Predicted Daily Use",
        f"{predictions.mean():,.1f} kWh"
    )
    st.metric(
        "Estimated Forecast Cost",
        f"₹{predictions.sum() * rate:,.0f}"
    )

    fig = px.line(
        forecast,
        x="Date",
        y="Predicted_kWh",
        markers=True,
        title="Forecast Energy Consumption"
    )
    st.plotly_chart(fig, width="stretch")

    st.warning(
        "Prototype forecast: this model uses simulated data and "
        "a simple time index. Validate it on real historical data "
        "before using its predictions for decisions."
    )

# Digital Twin Simulator
elif page == "Digital Twin Simulator":
    st.subheader("🧪 Digital Twin — What-if Simulator")
    st.write(
        "Change operating hours to estimate potential energy savings."
    )

    ac_hours = st.slider(
        "AC operating hours per day",
        0, 16, 8
    )
    light_hours = st.slider(
        "Lighting hours per day",
        0, 16, 8
    )
    computer_hours = st.slider(
        "Computer operating hours per day",
        0, 16, 7
    )

    ac_power = 1.5
    lights_power = 0.8
    computers_power = 2.0

    baseline = (
        8 * ac_power +
        10 * lights_power +
        8 * computers_power
    )

    simulated = (
        ac_hours * ac_power +
        light_hours * lights_power +
        computer_hours * computers_power
    )

    savings = baseline - simulated
    daily_saving = savings * rate

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Daily Use", f"{baseline:.1f} kWh")
    c2.metric("Simulated Daily Use", f"{simulated:.1f} kWh")
    c3.metric(
        "Estimated Daily Cost Change",
        f"₹{daily_saving:,.2f}",
        delta=f"{-savings:.1f} kWh"
    )

    monthly_saving = daily_saving * 30
    st.metric(
        "Estimated Monthly Savings",
        f"₹{monthly_saving:,.2f}"
    )

    if savings > 0:
        st.success(
            "The selected schedule uses less energy than the baseline."
        )
    elif savings < 0:
        st.warning(
            "The selected schedule uses more energy than the baseline."
        )
    else:
        st.info("The selected schedule matches the baseline.")

    st.caption(
        "Illustrative calculation only. Power ratings and baseline "
        "hours are assumptions, not measurements from real equipment."
    )

# Sustainability Report
elif page == "Sustainability Report":
    st.subheader("🌱 Sustainability Report")

    total = df["Energy_kWh"].sum()
    emission_factor = st.number_input(
        "Assumed CO₂ factor (kg per kWh)",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    emissions = total * emission_factor

    c1, c2, c3 = st.columns(3)
    c1.metric("Energy Used", f"{total:,.0f} kWh")
    c2.metric("Estimated CO₂", f"{emissions:,.0f} kg")
    c3.metric("Estimated Cost", f"₹{total * rate:,.0f}")

    st.markdown("### Room-wise Energy Summary")
    summary = (
        df.groupby("Room", as_index=False)
        .agg(
            Total_kWh=("Energy_kWh", "sum"),
            Average_kWh=("Energy_kWh", "mean")
        )
    )
    summary["Estimated_Cost_Rs"] = summary["Total_kWh"] * rate
    st.dataframe(summary, width="stretch")

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Sustainability CSV",
        data=csv,
        file_name="EnergyTwin_Sustainability_Report.csv",
        mime="text/csv"
    )

    st.caption(
        "CO₂ emissions are estimates based on the selected factor. "
        "Use an appropriate local electricity-grid factor for a real report."
    )