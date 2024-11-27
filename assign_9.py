import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = 'C:\\Users\\Devavrat Tapare\\Downloads\\3rd YEAR\\DMDW Lab\\sample_product_data.csv'
data = pd.read_csv(file_path)

# Set a threshold for "high popularity" (Popularity > 50 is classified as '1', otherwise '0')
data['HighPopularity'] = (data['Popularity'] > 50).astype(int)

# Select features and target variable
X = data[['Price', 'Rating', 'Stock', 'Weight']]  # Feature columns
y = data['HighPopularity']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Probabilities for each prediction
y_proba = model.predict_proba(X_test)
print("Predicted Probabilities:")
print(y_proba)
