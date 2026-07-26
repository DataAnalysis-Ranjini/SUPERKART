import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://backend:7860"

st.title("SuperKart Sales Prediction App")

# Selection for both modes
mode = st.sidebar.selectbox("Select Mode", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    st.header("Individual Item Prediction")
    weight = st.number_input("Product Weight", value=12.0)
    mrp = st.number_input("Product MRP", value=115.0)
    age = st.number_input("Store Age (Years)", value=15)

    # User input for Perishable / Non Perishable
    product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": weight,
            "Product_MRP": mrp,
            "Store_Age": age,
            "Product_Sugar_Content": "Low Fat",
            "Product_Allocated_Area": 0.05,
            "Store_Size": "Medium",
            "Store_Location_City_Type": "Tier 2",
            "Store_Type": "Supermarket Type1",
            "Product_Category_Code": "FD",
            "Product_Category_Prefix": "FD",
            "Store_Id": "OUT049",
            "Product_Type_Category": product_type_category
        }

        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
        if response.status_code == 200:
            pred = response.json()['Product_Store_Sales_Total']
            st.success(f"Estimated Sales: ${pred:.2f}")
        else:
            st.error("Unable to connect to the prediction API.")

elif mode == "Batch Prediction":
    st.header("Batch File Prediction")
    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Run Batch Predictions"):
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})
            if response.status_code == 200:
                predictions = response.json()
                st.success("Batch Predictions Complete!")
                st.write(predictions)
            else:
                st.error("Unable to connect to the prediction API.")
