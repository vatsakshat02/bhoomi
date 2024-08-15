import tensorflow as tf
import pickle
import numpy as np

model = tf.keras.models.load_model('crop_recommendation_model4.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def get_crop_recommendations(temperature, moisture, nitrogen, phosphorus, potassium, pH):
    
    new_data = np.array([[temperature, moisture, nitrogen, phosphorus, potassium, pH]])
    
  
    predictions = model.predict(new_data)
    predicted_crops = label_encoder.inverse_transform(predictions.argsort()[0][::-1])
    
    
    recommendations = []
    for i, crop in enumerate(predicted_crops):
        recommendations.append((crop, predictions[0][predictions.argsort()[0][::-1][i]]))
    
    return recommendations

if __name__ == "__main__":
   
    temperature = 23
    moisture = 50
    nitrogen = 14
    phosphorus = 20
    potassium = 25
    pH = 6.5
    
    recommendations = get_crop_recommendations(temperature, moisture, nitrogen, phosphorus, potassium, pH)
    
    
    print("Crop Recommendations:")
    for crop, probability in recommendations:
        print(f"{crop}: {probability:.4f}")
