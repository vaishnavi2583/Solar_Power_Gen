import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Solar Power Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {text-align:center; color:#ff9800;}
</style>
""", unsafe_allow_html=True)

st.title("Solar Power Prediction Dashboard")
st.write("Predict solar power output using ML and weather inputs.")
st.markdown("---")

model = pickle.load(open("gbr.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.sidebar.header("Input Parameters")

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 30.0, 5.0)

distance_to_noon = st.sidebar.number_input("Distance to Solar Noon", 0.0, 1.0, 0.5)
wind_direction = st.sidebar.number_input("Wind Direction (°)", 0.0, 360.0, 180.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Summary")
    st.table({
        "Temperature (°C)": [temperature],
        "Humidity (%)": [humidity],
        "Wind Speed (m/s)": [wind_speed],
        "Distance to Noon": [distance_to_noon],
        "Wind Direction (°)": [wind_direction]
    })

with col2:
    st.subheader("Prediction Output")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Predict Solar Power Output"):

        input_data = np.array([[temperature, humidity, wind_speed,
                                distance_to_noon, wind_direction]])

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        st.metric("Predicted Solar Power Output", f"{prediction:.2f} kW")

        st.subheader("Prediction Visualization")
        st.bar_chart({"Predicted Output (kW)": [prediction]})

        st.session_state.history.append(prediction)
        st.write("Past Predictions:", st.session_state.history)

        report = pd.DataFrame({
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Wind Speed": [wind_speed],
            "Distance to Noon": [distance_to_noon],
            "Wind Direction": [wind_direction],
            "Predicted Output (kW)": [prediction]
        })

        st.download_button(
            "Download Report",
            report.to_csv(index=False),
            "solar_report.csv"
        )

st.subheader("Monthly Output Trend Example")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
output = [120, 140, 160, 180, 210, 250]

plt.figure()
plt.plot(months, output)
st.pyplot(plt)

st.markdown("---")
st.caption("Developed by Vaishnavi | Solar Power Prediction System ☀")
