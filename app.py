import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("superkart_random_forest_model_v1_0.joblib")

model = load_model()

st.title("SuperKart Sales Prediction App")

mode = st.sidebar.selectbox("Select Mode", ["Single Prediction", "Batch Prediction"])

# Define exact types for every single column matching your model
NUMERIC_COLS = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Establishment_Year', 'Store_Age', 'Product_Type_Category'
]

CATEGORICAL_COLS = [
    'Product_Sugar_Content', 'Product_Type', 'Store_Size',
    'Store_Location_City_Type', 'Store_Type', 'Product_Category_Code',
    'Product_Category_Prefix', 'Store_Id'
]

EXPECTED_COLUMNS = NUMERIC_COLS + CATEGORICAL_COLS

if mode == "Single Prediction":
    st.subheader("Single Product Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", value=12.5)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Fat", "Regular", "Non Fat"])
        product_allocated_area = st.number_input("Product Allocated Area", value=0.05)
        product_type = st.text_input("Product Type", "Dairy")
        product_mrp = st.number_input("Product MRP", value=150.0)
        store_establishment_year = st.number_input("Store Establishment Year", value=2005, step=1)
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        
    with col2:
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
        product_category_code = st.text_input("Product Category Code", "FD")
        store_age = st.number_input("Store Age", value=15, step=1)
        product_type_category = st.number_input("Product Type Category", value=1, step=1)
        product_category_prefix = st.text_input("Product Category Prefix", "FD")
        store_id = st.text_input("Store Id", "OUT049")

    if st.button("Predict Sales"):
        sample = {
            'Product_Weight': product_weight,
            'Product_Sugar_Content': product_sugar_content,
            'Product_Allocated_Area': product_allocated_area,
            'Product_Type': product_type,
            'Product_MRP': product_mrp,
            'Store_Establishment_Year': store_establishment_year,
            'Store_Size': store_size,
            'Store_Location_City_Type': store_location_city_type,
            'Store_Type': store_type,
            'Product_Category_Code': product_category_code,
            'Store_Age': store_age,
            'Product_Type_Category': product_type_category,
            'Product_Category_Prefix': product_category_prefix,
            'Store_Id': store_id
        }
        
        input_data = pd.DataFrame([sample])
        input_data = input_data.reindex(columns=EXPECTED_COLUMNS)
        
        # Enforce strict type conversion safely to prevent ufunc / isnan errors
        for col in NUMERIC_COLS:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0.0)
            
        for col in CATEGORICAL_COLS:
            input_data[col] = input_data[col].fillna("Missing").astype(str).str.strip()

        try:
            prediction = model.predict(input_data)[0]
            predicted_sales = round(float(prediction), 2)
            st.success(f"Predicted Total Store Sales: **${predicted_sales:,.2f}**")
        except Exception as err:
            st.error(f"Prediction error: {err}")

elif mode == "Batch Prediction":
    st.subheader("Batch Prediction via CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        input_data.columns = input_data.columns.str.strip()
        
        for col in EXPECTED_COLUMNS:
            if col not in input_data.columns:
                input_data[col] = "Missing" if col in CATEGORICAL_COLS else 0.0

        input_data = input_data[EXPECTED_COLUMNS]
        
        for col in NUMERIC_COLS:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0.0)
            
        for col in CATEGORICAL_COLS:
            input_data[col] = input_data[col].fillna("Missing").astype(str).str.strip()
            
        try:
            predictions = model.predict(input_data).tolist()
            rounded_predictions = [round(float(pred), 2) for pred in predictions]
            
            input_data['Predicted_Sales'] = rounded_predictions
            st.write("### Prediction Results Preview:")
            st.dataframe(input_data, height=400)
            
            csv_data = input_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name="superkart_predictions.csv",
                mime="text/csv",
            )
        except Exception as err:
            st.error(f"Batch prediction error: {err}")
