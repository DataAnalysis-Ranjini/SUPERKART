import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
model = joblib.load("superkart_random_forest_model_v1_0.joblib")

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()

    # Mapping exact columns from your dataset
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_Type': data['Product_Type'],
        'Product_MRP': data['Product_MRP'],
        'Store_Establishment_Year': data['Store_Establishment_Year'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Category_Code': data['Product_Category_Code'],
        'Store_Age': data['Store_Age'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    input_data = pd.DataFrame([sample])
    prediction = model.predict(input_data)[0]
    predicted_sales = round(float(prediction), 2)

    return jsonify({'Product_Store_Sales_Total': predicted_sales})

@superkart_api.post('/v1/predictbatch')
def predict_batch():
    file = request.files['file']
    input_data = pd.read_csv(file)

    predictions = model.predict(input_data).tolist()
    rounded_predictions = [round(float(pred), 2) for pred in predictions]

    product_ids = input_data['Product_Id'].tolist() if 'Product_Id' in input_data.columns else list(range(len(predictions)))
    output_dict = dict(zip(product_ids, rounded_predictions))

    return output_dict

if __name__ == '__main__':
    superkart_api.run(debug=True)
