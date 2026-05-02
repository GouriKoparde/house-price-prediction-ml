import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# -------------------------------
# Step 1: Dataset
# -------------------------------

np.random.seed(42)

data = pd.DataFrame({
    'area': np.random.randint(500, 5000, 200),
    'bedrooms': np.random.randint(1, 6, 200),
    'bathrooms': np.random.randint(1, 4, 200),
    'age': np.random.randint(0, 30, 200),
    'parking': np.random.randint(0, 2, 200)
})

# Generate realistic price
data['price'] = (
    data['area'] * 3000 +
    data['bedrooms'] * 500000 +
    data['bathrooms'] * 300000 -
    data['age'] * 10000 +
    data['parking'] * 200000 +
    np.random.randint(-200000, 200000, 200)
)

# -------------------------------
# Step 2: Visualization
# -------------------------------

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")

# Scatter plot (area vs price)
plt.figure()
plt.scatter(data['area'], data['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.savefig("outputs/area_vs_price.png")

# -------------------------------
# Step 3: Model
# -------------------------------

X = data[['area', 'bedrooms', 'bathrooms', 'age', 'parking']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "models/house_price_model.pkl")
print("Model saved successfully!")


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
# -------------------------------
# Random Forest Model
# -------------------------------

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# -------------------------------
# Step 4: Prediction
# -------------------------------

new_house = pd.DataFrame({
    'area': [2200],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [4],
    'parking': [1]
})
prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0])