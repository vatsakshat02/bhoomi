import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

data = pd.read_csv('accurate_crop_recommendation_with_disease2.csv')

X = data[['Temperature', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH']]
y = data['Crop']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Dense(16, input_dim=6, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('crop_recommendation_model4.h5')
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
