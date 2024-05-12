import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the data
data = pd.read_csv('/content/Customer_Data (1).csv')

# Select the relevant attributes
attributes = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
              'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
              'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
              'PURCHASES_INSTALLMENTS_FREQUENCY']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(data[attributes])

# Train the KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Add the cluster labels to the data
data['cluster_kmeans'] = kmeans.labels_

# Train the hierarchical clustering model with the optimal number of clusters
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical.fit(X)
data['cluster_hierarchical'] = hierarchical.labels_

# Train the DBSCAN model with the optimal parameters
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan.fit(X)

# Check if the DBSCAN algorithm found any clusters
if np.any(dbscan.labels_ != -1):
    data['cluster_dbscan'] = dbscan.labels_
else:
    print("DBSCAN found no clusters.")

# Calculate the frequency lift for each cluster
data['frequency_lift'] = data['PURCHASES_FREQUENCY'] / data.groupby('cluster_kmeans')['PURCHASES_FREQUENCY'].transform('mean')

# Select the cluster with the highest frequency lift
best_cluster = data.groupby('cluster_kmeans')['frequency_lift'].mean().idxmax()

# Select the customers in the best cluster
best_customers = data[data['cluster_kmeans'] == best_cluster]

# Plot the scatter plot for the KMeans model
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.xlabel('BALANCE')
plt.ylabel('PURCHASES')
plt.title('KMeans Clustering')
plt.show()

# Plot the dendrogram for the hierarchical clustering model
fig = plt.figure(figsize=(10, 5))
dn = dendrogram(linkage(X, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Plot the scatter plot for the DBSCAN model
if np.any(dbscan.labels_ != -1):
    plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
    plt.xlabel('BALANCE')
    plt.ylabel('PURCHASES')
    plt.title('DBSCAN Clustering')
    plt.show()

# Print the results
print("Best cluster:", best_cluster)
print("Average frequency in best cluster:", data.loc[data['cluster_kmeans'] == best_cluster, 'PURCHASES_FREQUENCY'].mean())
print("Number of customers in best cluster:", len(best_customers))
print("Total frequency in best cluster:", best_customers['PURCHASES_FREQUENCY'].sum())
print("Total frequency lift in best cluster:", best_customers['frequency_lift'].sum())
