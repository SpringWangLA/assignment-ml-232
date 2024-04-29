from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = np.array([
  [1, 1, 1],
  [1, 2, 1],
  [1, 3, 1],
  [2, 1, 1],
  [2, 2, 1],
  [2, 3, 1],
  [2, 3.5, 1],
  [2.5, 2, 1],
  [3.5, 1, 1],
  [3.5, 2, 1],
  [3.5, 3, 2],
  [3.5, 4, 2],
  [4.5, 1, 2],
  [4.5, 2, 2],
  [4.5, 3, 2],
  [5, 4, 2],
  [5, 5, 2],
  [6, 3, 2],
  [6, 4, 2],
  [6, 5, 2]
])

# Assuming the data is stored in a NumPy array 'data' with features in the first 3 columns and class label in the 4th column
features = data[:, :2]  # Select features (first 3 columns)
labels = data[:, -1]    # Extract class labels (last column)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction (e.g., reduce to 2 components)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)

# Train a KNN classifier on the transformed features and class labels
knn = KNeighborsClassifier()
knn.fit(pca_data, labels)