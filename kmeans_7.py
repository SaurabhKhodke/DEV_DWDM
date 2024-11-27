import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("student_performance_data.csv")
print(df)

# Convert categorical variables to dummy/indicator variables
df1 = pd.get_dummies(df)
print(df1)

# Display info about the DataFrame
df.info()

# Scatter plot before scaling
sns.scatterplot(x=df['Age'], y=df['StudyHoursPerWeek'])
plt.title('Scatter Plot Before Scaling')
plt.show()

# Apply StandardScaler to 'Age' and 'StudyHoursPerWeek' columns
std_scalar = StandardScaler()
scaled_values = std_scalar.fit_transform(df[['Age', 'StudyHoursPerWeek']])

# Convert the scaled array back to a DataFrame for further use
scaled_df = pd.DataFrame(scaled_values, columns=['Age', 'StudyHoursPerWeek'])

# Scatter plot after scaling
sns.scatterplot(x=scaled_df['Age'], y=scaled_df['StudyHoursPerWeek'])
plt.title('Scatter Plot After Scaling')
plt.show()

# Apply KMeans clustering
kmean_model = KMeans(n_clusters=3)
y_pred = kmean_model.fit_predict(scaled_df)  # Use scaled_df for clustering

# Scatter plot with KMeans clustering results
plt.scatter(x=scaled_df['Age'], y=scaled_df['StudyHoursPerWeek'], c=y_pred, cmap='rainbow')
plt.title('KMeans Clustering Results')
plt.xlabel('Age')
plt.ylabel('Study Hours Per Week')
plt.show()
