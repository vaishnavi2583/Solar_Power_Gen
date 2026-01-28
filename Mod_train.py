#!/usr/bin/env python
# coding: utf-8

# In[87]:


import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import requests


# In[88]:


st.set_page_config("Solar Power Dashboard", layout="wide")


# In[89]:


st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {text-align:center; color:#ff9800;}
</style>
""", unsafe_allow_html=True)


# In[90]:


st.title("Solar Power Prediction Dashboard")
st.write("Predict solar power output using ML and weather inputs.")
st.markdown("---")


# In[91]:


# Load trained model and scaler
model = pickle.load(open("gbr.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# In[102]:


##Input Section
st.sidebar.header("Input Parameters")
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
wind_speed = st.sidebar.number_input("🌬 Wind Speed (m/s)", 0.0, 30.0, 5.0)
distance_to_noon = st.sidebar.number_input("Distance to Solar Noon", 0.0, 1.0, 0.5)
wind_direction = st.sidebar.number_input("Wind Direction (°)", 0.0, 360.0, 180.0)


# In[96]:


# Layout Columns
col1, col2 = st.columns(2)


# In[97]:


col1, col2 = st.columns(2)


# In[106]:


# INPUT SUMMARY
with col1:
    st.subheader("📊 Input Summary")
    st.table({
        "Temperature (°C)": [temperature],
        "Humidity (%)": [humidity],
        "Wind Speed (m/s)": [wind_speed],
        "Distance to Noon": [distance_to_noon],
        "Wind Direction (°)": [wind_direction]
    })


# In[109]:


# Prediction button
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
    st.download_button("Download Report", report.to_csv(index=False), "solar_report.csv")


# In[110]:


# Monthly Trend Graph
st.subheader("Monthly Output Trend Example")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
output = [120, 140, 160, 180, 210, 250]
plt.figure()
plt.plot(months, output)
st.pyplot(plt)


# In[111]:


st.markdown("---")
st.caption("Developed by Vaishnavi | Solar Power Prediction System")


# In[ ]:




