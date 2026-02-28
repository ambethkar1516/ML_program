import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Create a dummy dataset (In a real scenario, use pd.read_csv('weather.csv'))
data = {
    'MinTemp': [15, 17, 18, 20, 22, 25, 28, 30, 12, 10],
    'MaxTemp': [25, 28, 30, 32, 35, 38, 42, 45, 20, 18]
}
df = pd.DataFrame(data)

# 2. Reshape data for Scikit-Learn (needs 2D arrays)
X = df['MinTemp'].values.reshape(-1, 1)
y = df['MaxTemp'].values.reshape(-1, 1)

# 3. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Visualize the Results
plt.scatter(X_train, y_train, color='blue', label='Actual Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
plt.title('Min Temp vs Max Temp')
plt.xlabel('Min Temperature')
plt.ylabel('Max Temperature')
plt.legend()
plt.show()

# 7. Evaluate the Model
print(f"Prediction for a 21°C Min Temp: {model.predict([[21]])[0][0]:.2f}°C")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))