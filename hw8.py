# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:20:07 2024

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Create a DataFrame for easier visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Visualize pairplot to see the distribution
sns.pairplot(df, hue='species', palette='viridis')
plt.title('Iris Dataset Pairplot')
plt.show()

# Standardize the feature data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Initialize and fit a GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_std)

# Predict cluster assignments
clusters = gmm.predict(X_std)

# Plot the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X_std[:, 0], X_std[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('GMM Clustering of Iris Dataset')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.colorbar(label='Cluster')
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_std, clusters)
print(f'Silhouette Score: {silhouette_avg:.2f}')

