# Step 1: Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Data Collection
data = pd.read_csv('C:\\Users\\Devavrat Tapare\\Downloads\\3rd YEAR\\DMDW Lab\\gym_membership.csv')  # Load your dataset here
data = data.fillna(0)  # Fill missing values with 0 or an appropriate method

# Step 3: Exploratory Data Analysis (EDA)
print(data.head())  # Show the first few rows of the dataset
print(data.info())  # Get info on the data types and missing values
print(data.describe())  # Get statistical summary of the dataset

# Step 4: Feature Engineering
# Convert categorical variables to numerical if needed
data['personal_training'] = data['personal_training'].astype(int)  # Convert 'personal_training' to numeric (0, 1)
data['attend_group_lesson'] = data['attend_group_lesson'].apply(lambda x: 1 if x == 'TRUE' else 0)  # Convert to binary
data['avg_time_check_in'] = pd.to_timedelta(data['avg_time_check_in']).dt.total_seconds() / 60  # Convert time to minutes
data['avg_time_check_out'] = pd.to_timedelta(data['avg_time_check_out']).dt.total_seconds() / 60  # Convert time to minutes
data['avg_time_in_gym'] = pd.to_timedelta(data['avg_time_in_gym']).dt.total_seconds() / 60  # Convert time to minutes

# We will drop non-numerical columns that won't be used in prediction
data = data.drop(columns=['id', 'first_name', 'gender', 'abonoment_type', 'fav_group_lesson', 'visit_per_week', 'days_per_week'])

# Target variable: Let's assume 'personal_training' indicates whether they are likely to show up
y = data['personal_training']
X = data.drop('personal_training', axis=1)  # All other columns as features

# Step 5: Splitting the dataset into training and testing sets (Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% testing

# Step 6: Model Training
model = GaussianNB()  # Create an instance of the Gaussian Naive Bayes model
model.fit(X_train, y_train)  # Fit the model on training data

# After the data processing steps and before the prediction

# Step 7: Model Evaluation
y_pred = model.predict(X_test)  # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
conf_matrix = confusion_matrix(y_test, y_pred)  # Get confusion matrix
class_report = classification_report(y_test, y_pred)  # Get classification report

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Check the columns of X
print("Columns in feature set (X):", X.columns)
print("Number of columns in feature set (X):", len(X.columns))

# Visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Did not show up', 'Showed up'], 
            yticklabels=['Did not show up', 'Showed up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Single User Input Testing
# Create a function for predicting a single user input
def predict_user_input(input_data):
    # Ensure input data matches the expected format (DataFrame)
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)  # Predict using the trained model
    return "Showed Up" if prediction[0] == 1 else "Did Not Show Up"

# Example of user input
# Adjust the user input to match the number of features in X
user_input = [30, 130, 160, 75, 1, 0]  # Example input; change as needed
result = predict_user_input(user_input)
print(f'Prediction for user input {user_input}: {result}')

