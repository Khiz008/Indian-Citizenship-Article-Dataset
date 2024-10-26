import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data_path = 'C:\\Users\\92317\\Pictures\\CAA 2019\\sentiment_analysis_results.csv'
data = pd.read_csv(data_path)

# Define the features and target variable
features = ['Polarity', 'Subjectivity', 'Media Coverage', 'Government Statements and Actions', 
             'Legal Challenges and Court Rulings', 'Public Protests and Demonstrations']
target = 'Public Sentiment'

# Prepare the data
X = data[features]
y = data[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Make predictions
y_pred = model.predict(X_scaled)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Display results
print("Linear Regression Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.xlabel('Actual Public Sentiment')
plt.ylabel('Predicted Public Sentiment')
plt.title('Actual vs Predicted Public Sentiment')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.show()

# Correlation matrix
correlat
