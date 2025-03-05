#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Function to load dataset
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None


# In[4]:


# Title of the App
st.title("🚚 Delivery Delay Prediction System")

# Sidebar - File Upload
st.sidebar.header("C://Users/saura/Downloads/Processed_Sales_Data.xls")
sales_file = st.sidebar.file_uploader("C://Users/saura/Downloads/Sales_Data.csv", type=["csv"])
iod_file = st.sidebar.file_uploader("C://Users/saura/Downloads/IOD.csv", type=["csv"])
obd_file = st.sidebar.file_uploader("C://Users/saura/Downloads/OBD.csv", type=["csv"])


# In[5]:


# Process files if uploaded
if sales_file and iod_file and obd_file:
    sales_data = load_data(sales_file)
    iod_data = load_data(iod_file)
    obd_data = load_data(obd_file)

    st.write("### Preview of Sales Data:")
    st.dataframe(sales_data.head())

    # Merge datasets
    merged_sales_iod = sales_data.merge(iod_data, on="Invoice_No", how="left")
    merged_sales_iod_obd = merged_sales_iod.merge(obd_data, on="OBD_No", how="left")

    # Feature Engineering
    merged_sales_iod_obd["Delivery_Delayed"] = merged_sales_iod_obd["Delivery_Days"].apply(lambda x: 1 if x > 5 else 0)

    # Select features for model training
    features = ["Supplying_Plant_Code", "Sales_Office_ID", "QTY", "Sales_Value"]
    target = "Delivery_Delayed"

    # Handle missing values
    merged_sales_iod_obd[features] = merged_sales_iod_obd[features].fillna(0)

    # Split data
    X = merged_sales_iod_obd[features]
    y = merged_sales_iod_obd[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save Model
    joblib.dump(model, "delay_prediction_model.pkl")

    st.write("### ✅ Model Training Completed Successfully!")

    # Sidebar - Predict Delay
    st.sidebar.header("Predict Delivery Delay")
    qty = st.sidebar.number_input("Quantity Ordered", min_value=1, max_value=1000, value=10)
    sales_value = st.sidebar.number_input("Sales Value", min_value=100, max_value=100000, value=5000)
    plant_code = st.sidebar.number_input("Supplying Plant Code", min_value=100, max_value=1000, value=200)
    sales_office = st.sidebar.number_input("Sales Office ID", min_value=100, max_value=1000, value=300)

    if st.sidebar.button("Predict Delay"):
        model = joblib.load("delay_prediction_model.pkl")
        new_data = [[plant_code, sales_office, qty, sales_value]]
        prediction = model.predict(new_data)
        result = "🚨 Delayed" if prediction[0] == 1 else "✅ On-Time"
        st.write(f"### Prediction: {result}")


# In[ ]:




