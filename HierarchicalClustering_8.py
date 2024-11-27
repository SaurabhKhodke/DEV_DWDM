import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
customer_data = pd.read_csv('student_performance_data.csv')

# Check the shape of the dataset (rows, columns)
print(customer_data.shape)

# Display the first few rows of the dataset
print(customer_data.head())

print('---------------------------------------------')

# Select the columns for clustering (ensure they are numeric)
# Here we're including more features for better clustering (e.g., GPA, StudyHoursPerWeek, AttendanceRate)
data = customer_data[['StudyHoursPerWeek', 'AttendanceRate', 'GPA']].values
print("Selected Data for Clustering:")
print(data)

# Step 1: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 2: Plot the Dendrogram
plt.figure(figsize=(10, 7))
plt.title("Student Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

# Show where we might cut the dendrogram for cluster assignment
plt.axhline(y=5, color='r', linestyle='--')  # Adjust this line to fine-tune the number of clusters
plt.show()

# Step 3: Perform Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
labels_ = cluster.fit_predict(data_scaled)

print('---------------------------------------------')
print('Cluster Labels:')
print(labels_)

# Step 4: Use PCA for Dimensionality Reduction (2D for visualization)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Step 5: Plot the clustered data using the first two principal components
plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster.labels_, cmap='rainbow')
plt.title('Clustered Data (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
