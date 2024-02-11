# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:22:36 2024

@author: Aaron Desktop
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_excel('regime.xlsx')
# Normalize the data excluding 'Date'
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df.drop('Date', axis=1)), columns=df.columns[1:])
df_normalized['Date'] = df['Date']

# Preparing data for clustering (excluding 'Date')
data_for_clustering = df_normalized.drop('Date', axis=1).values

# Finding the optimal number of clusters using the Elbow Method
inertia = []  # Sum of squared distances of samples to their closest cluster center
silhouette_scores = []  # Silhouette Coefficient score
K_range = range(2, 11)  # Considering 2 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data_for_clustering)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_for_clustering, kmeans.labels_))

# Plotting the Elbow Method result

'''
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', color='r')
plt.title('Silhouette Score For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
'''

# Applying K-Means clustering with the chosen number of clusters
chosen_k = 4
kmeans_final = KMeans(n_clusters=chosen_k, random_state=42).fit(data_for_clustering)

# Adding the cluster labels to the dataframe
df_normalized['Cluster'] = kmeans_final.labels_

pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
data_reduced = pca.fit_transform(data_for_clustering)

plt.figure(figsize=(8, 6))
for i in range(chosen_k):
    plt.scatter(data_reduced[kmeans_final.labels_ == i, 0], data_reduced[kmeans_final.labels_ == i, 1], label=f'Cluster {i}')

'''
plt.title('Data Points Clustered into 4 Groups')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
'''

# Applying K-Means clustering with 4 clusters (as previously determined)
chosen_k = 4
kmeans_final = KMeans(n_clusters=chosen_k, random_state=42).fit(df_normalized.drop('Date', axis=1))

# Adding the cluster labels to the dataframe
df_normalized['Cluster'] = kmeans_final.labels_

# Creating a new dataframe with Date and Cluster Number columns
df_clusters = df_normalized[['Date', 'Cluster']].copy()

# Display the first few rows of the new dataframe
#df_clusters.head()

# Applying K-Means clustering with 4 clusters (as previously determined)

# Creating a new dataframe with Date and Cluster Number columns
df_clusters = df_normalized[['Date', 'Cluster']].copy()

df_clusters_sorted = df_clusters.sort_values(by='Date')

# Plotting
plt.figure(figsize=(12, 6))
colors = ['blue', 'green', 'red', 'purple']  # Distinct colors for each cluster

for cluster in df_clusters_sorted['Cluster'].unique():
    # Filter data for each cluster
    cluster_data = df_clusters_sorted[df_clusters_sorted['Cluster'] == cluster]
    plt.plot(cluster_data['Date'], cluster_data['Cluster'], label=f'Cluster {cluster}', marker='o', linestyle='', color=colors[cluster])

'''
plt.title('Time Series of Clusters')
plt.xlabel('Date')
plt.ylabel('Cluster')
plt.yticks(df_clusters_sorted['Cluster'].unique())  # Ensure y-axis only shows the cluster numbers
plt.legend()
plt.tight_layout()
plt.show()
'''


# Calculate the mean of each variable within each cluster
cluster_means = df.groupby('Cluster').mean()

# Display the mean values for interpretation

df['Cluster'] = kmeans_final.labels_

# Calculate the mean of each variable within each cluster
cluster_means = df.groupby('Cluster').mean()

# Display the mean values for interpretation
# Mapping cluster numbers to the new cluster names
cluster_names = {
    0: 'Recessionary Conditions',
    1: 'Stable Growth',
    2: 'Inflationary Pressure',
    3: 'High Growth'
}

# Apply the mapping to the 'Cluster' column to replace cluster numbers with names
df['Cluster Name'] = df['Cluster'].map(cluster_names)

# Ensure the 'Date' column is in datetime format for plotting
df['Date'] = pd.to_datetime(df['Date'])

# Sort the dataframe by date to ensure correct plotting order
df_sorted = df.sort_values(by='Date')

# Plotting with new cluster names
plt.figure(figsize=(12, 8))

# Unique colors for each cluster name
colors = {
    'High Growth': 'blue',
    'Stable Growth': 'green',
    'Inflationary Pressure': 'red',
    'Recessionary Conditions': 'purple'
}

for cluster_name, color in colors.items():
    cluster_data = df_sorted[df_sorted['Cluster Name'] == cluster_name]
    plt.plot(cluster_data['Date'], cluster_data['Cluster'], label=cluster_name, marker='o', linestyle='', color=color)

plt.title('Time Series of Economic Conditions')
plt.xlabel('Date')
plt.ylabel('Economic Condition')
plt.yticks(df_sorted['Cluster'].unique(), labels=df_sorted['Cluster Name'].unique())  # Ensure y-axis shows the new cluster names
plt.legend()
plt.tight_layout()
plt.show()
