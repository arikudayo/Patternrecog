import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Set the linkage method
linkage_method = 'single' 

# Perform hierarchical clustering
Z = linkage(iris_df, method=linkage_method)

# Create a dendrogram
plt.figure(figsize=(18, 14))
dendrogram(Z, labels=iris.target, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Species Index')
plt.ylabel('Distance')
plt.show()