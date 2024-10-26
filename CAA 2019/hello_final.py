import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
path = 'C:\\Users\\92317\\Pictures\\CAA 2019\\sentiment_analysis_results.csv'
data = pd.read_csv(path)

# Create a binary target variable for success/failure
# Assume 'Public Sentiment' > 5 indicates success (1), otherwise failure (0)
data['Success/Failure'] = data['Public Sentiment'].apply(lambda x: 1 if x > 5 else 0)

# Define features and target variable
features = ['Polarity', 'Subjectivity', 'Media Coverage', 
             'Government Statements and Actions', 
             'Legal Challenges and Court Rulings', 
             'Public Protests and Demonstrations']
X = data[features]
y = data['Success/Failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Failure', 'Success']
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Print text on confusion matrix plot
for i, j in enumerate(tick_marks):
    plt.text(j, i, conf_matrix[i, j], 
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
