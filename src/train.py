import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create dataset
np.random.seed(42)

data = pd.DataFrame({
    'area': np.random.randint(500, 5000, 200),
    'bedrooms': np.random.randint(1, 6, 200),
    'bathrooms': np.random.randint(1, 4, 200),
    'age': np.random.randint(0, 30, 200),
    'parking': np.random.randint(0, 2, 200)
})

data['price'] = (
    data['area'] * 3000 +
    data['bedrooms'] * 500000 +
    data['bathrooms'] * 300000 -
    data['age'] * 10000 +
    data['parking'] * 200000 +
    np.random.randint(-200000, 200000, 200)
)

# Features
X = data[['area', 'bedrooms', 'bathrooms', 'age', 'parking']]
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/house_price_model.pkl")
print("Model trained and saved!")

# Evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2:", r2_score(y_test, y_pred))