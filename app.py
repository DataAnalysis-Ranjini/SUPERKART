import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.title("SuperKart Sales Predictor")

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load('superkart_random_forest_model_v1_0.joblib')

model = load_model()

# Create tabs for Single vs Batch Prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

# ==========================================
# TAB 1: SINGLE PREDICTION FORM
# ==========================================
with tab1:
    st.header("Enter Product & Store Details")
    
    prod_weight = st.number_input("Product Weight", value=10.0)
    allocated_area = st.number_input("Product Allocated Area", value=0.1)
    mrp = st.number_input("Product MRP", value=150.0)
    establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2026, value=2010, step=1)

    current_year = datetime.now().year
    store_age = current_year - int(establishment_year)

    sugar_content = st.selectbox("Product Sugar Content", ["Low Fat", "Regular", "Non-Edible"])
    product_type = st.selectbox("Product Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods"])
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
    category_code = st.text_input("Product Category Code", value="DR")
    store_id = st.text_input("Store ID", value="OUT049")
    product_id = st.text_input("Product ID", value="FDA15")
    product_type_category = st.text_input("Product Type Category", value="1")

    product_category_prefix = str(product_id)[:2].upper()

    input_data = {
        'Product_Weight': float(prod_weight),
        'Product_Allocated_Area': float(allocated_area),
        'Product_MRP': float(mrp),
        'Store_Age': int(store_age),
        'Product_Sugar_Content': str(sugar_content),
        'Product_Type': str(product_type),
        'Store_Size': str(store_size),
        'Store_Location_City_Type': str(city_type),
        'Store_Type': str(store_type),
        'Product_Category_Code': str(category_code),
        'Store_Id': str(store_id),
        'Product_Id': str(product_id),
        'Product_Type_Category': str(product_type_category),
        'Product_Category_Prefix': str(product_category_prefix)
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Single Sales"):
        try:
            prediction = model.predict(input_df)
            st.success(f"Predicted Sales Total: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ==========================================
# TAB 2: BATCH PREDICTION (CSV UPLOAD)
# ==========================================
with tab2:
    st.header("Upload CSV for Batch Predictions")
    st.markdown("Your uploaded CSV must contain the required base columns.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", batch_df.head(10))
            
            if st.button("Run Batch Predictions"):
                # Clean column names (strip whitespace or handle lowercase/uppercase variants)
                batch_df.columns = batch_df.columns.str.strip()

                # Map alternative column names if they differ slightly in the uploaded CSV
                column_mapping = {
                    'Store_Establishment_Year': ['Store_Establishment_Year', 'Establishment_Year', 'Year'],
                    'Product_Weight': ['Product_Weight', 'Weight', 'Item_Weight'],
                    'Product_Allocated_Area': ['Product_Allocated_Area', 'Visibility', 'Item_Visibility', 'Allocated_Area'],
                    'Product_MRP': ['Product_MRP', 'MRP', 'Item_MRP'],
                    'Product_Sugar_Content': ['Product_Sugar_Content', 'Sugar_Content', 'Item_Fat_Content'],
                    'Product_Type': ['Product_Type', 'Item_Type'],
                    'Store_Size': ['Store_Size', 'Outlet_Size'],
                    'Store_Location_City_Type': ['Store_Location_City_Type', 'Outlet_Location_Type', 'City_Type'],
                    'Store_Type': ['Store_Type', 'Outlet_Type'],
                    'Product_Category_Code': ['Product_Category_Code', 'Category_Code'],
                    'Store_Id': ['Store_Id', 'Store_ID', 'Outlet_Identifier'],
                    'Product_Id': ['Product_Id', 'Product_ID', 'Item_Identifier'],
                    'Product_Type_Category': ['Product_Type_Category', 'Type_Category']
                }

                for target_col, variants in column_mapping.items():
                    if target_col not in batch_df.columns:
                        for v in variants:
                            if v in batch_df.columns:
                                batch_df[target_col] = batch_df[v]
                                break

                # Automatically engineer missing features safely
                if 'Store_Age' not in batch_df.columns:
                    if 'Store_Establishment_Year' in batch_df.columns:
                        batch_df['Store_Age'] = datetime.now().year - pd.to_numeric(batch_df['Store_Establishment_Year'], errors='coerce').fillna(2010)
                    else:
                        batch_df['Store_Age'] = 15

                if 'Product_Category_Prefix' not in batch_df.columns:
                    if 'Product_Id' in batch_df.columns:
                        batch_df['Product_Category_Prefix'] = batch_df['Product_Id'].astype(str).str[:2].str.upper()
                    else:
                        batch_df['Product_Category_Prefix'] = 'DR'

                # Explicit expected features list required by the model
                expected_cols = [
                    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age',
                    'Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type',
                    'Store_Type', 'Product_Category_Code', 'Store_Id', 'Product_Id',
                    'Product_Type_Category', 'Product_Category_Prefix'
                ]

                # Ensure all expected columns exist, filling missing ones with default safe placeholders
                for col in expected_cols:
                    if col not in batch_df.columns:
                        if col in ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age']:
                            batch_df[col] = 0.0
                        else:
                            batch_df[col] = 'Unknown'

                # Run predictions for all rows using strictly aligned columns
                predictions = model.predict(batch_df[expected_cols])
                batch_df['Predicted_Sales'] = predictions
                
                st.success("Batch predictions completed successfully!")
                
                # Display predictions for the first 10 items explicitly using head(10)
                st.subheader("Predictions for the first 10 items:")
                st.dataframe(batch_df.head(10), height=380)
                
                # Download button for the full results
                csv_output = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Predicted Results as CSV",
                    data=csv_output,
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")
