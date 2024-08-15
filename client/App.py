from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import json
from io import BytesIO

app = Flask(__name__)
CORS(app)  
try:
    disease_model = load_model('crop_disease_model.h5')
    crop_model = load_model('crop_recommendation_model.h5')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {str(e)}")


try:
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)
    print("Class labels loaded successfully.")
except Exception as e:
    print(f"Error loading class labels: {str(e)}")

index_to_crop = {
    0: 'Rice',
    1: 'Wheat',
    2: 'Maize',
    3: 'Cotton',
    4: 'Tea'
}

csv_file_path = 'accurate_crop_recommendation_with_disease2.csv'
try:
    data = pd.read_csv(csv_file_path)
    print("CSV data loaded successfully.")
except Exception as e:
    print(f"Error loading CSV data: {str(e)}")


def preprocess_image(image):
    img = image.resize((150, 150))  
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    print(f"Processed image array shape: {img_array.shape}")
    return img_array

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    try:
        img = load_img(BytesIO(file.read()))
        preprocessed_image = preprocess_image(img)
        print("Image preprocessed successfully.")

        predictions = disease_model.predict(preprocessed_image)
        print(f"Model raw predictions: {predictions}")

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        print(f"Predicted class index: {predicted_class_index}")
        
        if predicted_class_index < len(class_labels):
            predicted_class_label = class_labels[predicted_class_index]
        else:
            predicted_class_label = "Unknown Disease"
        print(f"Predicted class label: {predicted_class_label}")
        
        return jsonify({'disease': predicted_class_label})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Error during prediction'}), 500

@app.route('/recommend-crop', methods=['POST'])
def recommend_crop():
    data_input = request.json
    
    required_keys = ['Temperature', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH']
    if not all(key in data_input for key in required_keys):
        return jsonify({'error': f'Missing one or more required fields: {required_keys}'}), 400
    
    try:
        input_features = np.array([
            float(data_input['Temperature']),
            float(data_input['Moisture']),
            float(data_input['Nitrogen']),
            float(data_input['Phosphorus']),
            float(data_input['Potassium']),
            float(data_input['pH'])
        ]).reshape(1, -1)  
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    
    try:
        crop_prediction = crop_model.predict(input_features)
        print(f"Crop model predictions: {crop_prediction}")
    
        recommended_crop_index = np.argmax(crop_prediction, axis=1)[0]
        recommended_crop = index_to_crop.get(recommended_crop_index, "Unknown Crop")
        print(f"Recommended crop: {recommended_crop}")
    
        disease_row = data.loc[
            (data['Temperature'].astype(float).round(2) == float(data_input['Temperature'])) &
            (data['Moisture'].astype(float).round(2) == float(data_input['Moisture'])) &
            (data['Nitrogen'].astype(float).round(2) == float(data_input['Nitrogen'])) &
            (data['Phosphorus'].astype(float).round(2) == float(data_input['Phosphorus'])) &
            (data['Potassium'].astype(float).round(2) == float(data_input['Potassium'])) &
            (data['pH'].astype(float).round(2) == float(data_input['pH']))
        ]

        if not disease_row.empty:
            disease_present = disease_row.iloc[0]['Disease']
        else:
            disease_present = "No disease found"
        print(f"Soil disease: {disease_present}")

        return jsonify({'recommended_crop': recommended_crop, 'soil_disease': disease_present})

    except Exception as e:
        print(f"Error during crop recommendation or disease check: {str(e)}")
        return jsonify({'error': 'Error during crop recommendation or disease check'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
