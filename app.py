import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
model = joblib.load("superkart_random_forest_model_v1_0.joblib")

# Define exact expected columns to maintain consistency
EXPECTED_COLUMNS = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Establishment_Year', 'Store_Age', 'Product_Type_Category',
    'Product_Sugar_Content', 'Product_Type', 'Store_Size',
    'Store_Location_City_Type', 'Store_Type', 'Product_Category_Code',
    'Product_Category_Prefix', 'Store_Id'
]

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()

    # Mapping all 14 required columns securely from incoming JSON payload
    sample = {
        'Product_Weight': float(data.get('Product_Weight', 0.0)),
        'Product_Allocated_Area': float(data.get('Product_Allocated_Area', 0.0)),
        'Product_MRP': float(data.get('Product_MRP', 0.0)),
        'Store_Establishment_Year': int(data.get('Store_Establishment_Year', 2005)),
        'Store_Age': int(data.get('Store_Age', 15)),
        'Product_Type_Category': int(data.get('Product_Type_Category', 1)),
        'Product_Sugar_Content': str(data.get('Product_Sugar_Content', 'Low Fat')),
        'Product_Type': str(data.get('Product_Type', 'Dairy')),
        'Store_Size': str(data.get('Store_Size', 'Medium')),
        'Store_Location_City_Type': str(data.get('Store_Location_City_Type', 'Tier 2')),
        'Store_Type': str(data.get('Store_Type', 'Supermarket Type1')),
        'Product_Category_Code': str(data.get('Product_Category_Code', 'FD')),
        'Product_Category_Prefix': str(data.get('Product_Category_Prefix', 'FD')),
        'Store_Id': str(data.get('Store_Id', 'OUT049'))
    }

    input_data = pd.DataFrame([sample])
    input_data = input_data.reindex(columns=EXPECTED_COLUMNS)

    prediction = model.predict(input_data)[0]
    predicted_sales = round(float(prediction), 2)

    return jsonify({'Product_Store_Sales_Total': predicted_sales})

@superkart_api.post('/v1/predictbatch')
def predict_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found in the request'}), 400
        
    file = request.files['file']
    input_data = pd.read_csv(file)
    input_data.columns = input_data.columns.str.strip()

    # Ensure all expected columns are present for batch predictions
    for col in EXPECTED_COLUMNS:
        if col not in input_data.columns:
            input_data[col] = "Missing" if col in ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Product_Category_Code', 'Product_Category_Prefix', 'Store_Id'] else 0.0

    input_data = input_data[EXPECTED_COLUMNS]

    predictions = model.predict(input_data).tolist()
    rounded_predictions = [round(float(pred), 2) for pred in predictions]

    product_ids = input_data['Store_Id'].tolist() if 'Store_Id' in input_data.columns else list(range(len(predictions)))
    output_dict = dict(zip(product_ids, rounded_predictions))

    return jsonify(output_dict)

if __name__ == '__main__':
    superkart_api.run(debug=True)
