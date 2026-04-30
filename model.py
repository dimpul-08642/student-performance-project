import pandas as pd

# Load dataset
data = pd.read_csv("data/student.csv")

# Show first 5 rows
print("First 5 rows:\n")
print(data.head())

# Show basic info
print("\nDataset Info:\n")
print(data.info())

# Check missing values
print("\nMissing Values:\n")
print(data.isnull().sum())

# Show statistics
print("\nStatistics:\n")
print(data.describe())

# Select input features (X)
X = data[["hours_studied", "attendance", "previous_score", "sleep_hours"]]

# Select output (y)
y = data["final_score"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))

# -------------------------------
# LINEAR REGRESSION MODEL
# -------------------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression trained!")

# Predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

print("\nActual values:")
print(y_test.values)

# Evaluate Linear Regression
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("\nLinear Regression MAE:", mae)

# -------------------------------
# RANDOM FOREST MODEL
# -------------------------------
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
print("Random Forest MAE:", rf_mae)

# -------------------------------
# CHOOSE BEST MODEL
# -------------------------------
best_model = model if mae < rf_mae else rf_model

import pickle

# Save both models
with open("linear.pkl", "wb") as f:
    pickle.dump(model, f)

with open("rf.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\nBoth models saved!")

# -------------------------------
# VISUALIZATION
# -------------------------------
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()