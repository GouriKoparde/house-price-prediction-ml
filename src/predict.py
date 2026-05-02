import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Input: new house details
new_house = pd.DataFrame({
    'area': [2500],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [5],
    'parking': [1]
})

# Predict price
prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0])