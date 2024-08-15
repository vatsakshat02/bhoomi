import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import json

# Load the trained model
model = load_model('crop_disease_model.keras')

# Load class labels from the JSON file
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # Resize the image
    img_array = img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image
    return img_array

def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    
    predictions = model.predict(preprocessed_image)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label


image_path = r'C:\Users\Kunal\Documents\Bhoomi\download.jpg'
predicted_label = predict_disease(image_path)
print(f'The predicted disease is: {predicted_label}')
