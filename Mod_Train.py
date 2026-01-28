#!/usr/bin/env python
# coding: utf-8

# In[34]:


import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import requests


# In[65]:


st.set_page_config("Solar Power Dashboard", layout="wide")


# In[66]:


st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {text-align:center; color:#ff9800;}
</style>
""", unsafe_allow_html=True)


# In[67]:


st.title("Solar Power Prediction Dashboard")
st.write("Predict solar power output using ML and weather inputs.")
st.markdown("---")


# In[68]:


# Load trained model and scaler
model = pickle.load(open("gbr.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# In[69]:


st.sidebar.header("⚙ Input Parameters")
temperature = st.sidebar.number_input("🌡 Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.sidebar.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0)
wind_speed = st.sidebar.number_input("🌬 Wind Speed (m/s)", 0.0, 30.0, 5.0)


# In[27]:


##Input Section
distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon", value=0.5)
temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
wind_direction = st.sidebar.number_input("Wind Direction", value=180.0)
wind_speed = st.sidebar.number_input("Wind Speed", value=3.0)
sky_cover = st.sidebar.selectbox("Sky Cover (0-4)", [0,1,2,3,4])
visibility = st.sidebar.number_input("Visibility (km)", value=10.0)
humidity = st.sidebar.number_input("Humidity (%)", value=60.0)
average_wind_speed = st.sidebar.number_input("Average Wind Speed", value=3.2)
average_pressure = st.sidebar.number_input("Average Pressure", value=29.9)


# In[70]:


# Live Weather Option
st.sidebar.subheader("Live Weather Data")
use_live = st.sidebar.checkbox("Use Live Weather Data")


# In[71]:


if use_live:
    city = st.sidebar.text_input("City", "Bangalore")
    API_KEY = "YOUR_API_KEY_HERE"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        data = requests.get(url).json()
        temperature = data["main"]["temp"] - 273.15
        humidity = data["main"]["humidity"]
        st.sidebar.success("Live Weather Loaded")
    except:
        st.sidebar.error("API Error")


# In[72]:


# Layout Columns
col1, col2 = st.columns(2)


# In[43]:


col1, col2 = st.columns(2)


# In[73]:


# INPUT SUMMARY
with col1:
    st.subheader("Input Summary")
    st.table({
        "Temperature (°C)": [temperature],
        "Humidity (%)": [humidity],
        "Wind Speed (m/s)": [wind_speed]
    })


# In[80]:


# Prediction button
with col2:
    st.subheader("Prediction Output")
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.button("Predict Solar Power Output"):
        input_data = np.array([[temperature, humidity, wind_speed, distance_to_noon, wind_direction]])
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
            "Distance to Solar Noon": [distance_to_noon],
            "Wind Direction": [wind_direction],
            "Predicted Output (kW)": [prediction]
        })
        st.download_button(
            "Download Report",
            report.to_csv(index=False),
            "solar_report.csv"
        )


# In[81]:


# Monthly Trend Graph
st.subheader("Monthly Output Trend Example")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
output = [120, 140, 160, 180, 210, 250]
plt.figure()
plt.plot(months, output)
st.pyplot(plt)


# In[82]:


st.markdown("---")
st.caption("Developed by Vaishnavi | Solar Power Prediction System")

