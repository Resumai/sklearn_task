from sklearn.cluster import KMeans
 
# let's make some synthetic data blobs for clustering
from sklearn.datasets import make_blobs
 
def generate_data(n_samples=100, centers=3, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    return X, y
 
# # Generate synthetic data
X, y = generate_data(n_samples=500, centers=4)
# # plot the data
import matplotlib.pyplot as plt
# def plot_data(X, y=None):
#     plt.figure(figsize=(8, 6))
#     if y is not None:
#         plt.scatter(X[:, 0], X[:, 1])
#     else:
#         plt.scatter(X[:, 0], X[:, 1], marker='o')
#     plt.title('Synthetic Data Blobs')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.grid()
#     plt.show()
# plot_data(X, y)
# Apply KMeans clustering
kmeans = KMeans(n_clusters=3,n_init=10, random_state=42)
means = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
# Plot the clustered data
def plot_clustered_data(X, labels, centers):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x', s=200, label='Centroids')
    plt.title('KMeans Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()
print(means)
plot_clustered_data(X, means, centers)








# Second itteration of examples:

from sklearn.cluster import KMeans
 
# let's make some synthetic data blobs for clustering
from sklearn.datasets import make_blobs
 
def generate_data(n_samples=100, centers=3, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    return X, y
 
# # Generate synthetic data
X, y = generate_data(n_samples=500, centers=9, random_state=42)
# # plot the data
import matplotlib.pyplot as plt
# def plot_data(X, y=None):
#     plt.figure(figsize=(8, 6))
#     if y is not None:
#         plt.scatter(X[:, 0], X[:, 1])
#     else:
#         plt.scatter(X[:, 0], X[:, 1], marker='o')
#     plt.title('Synthetic Data Blobs')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.grid()
#     plt.show()
# plot_data(X, y)
# Apply KMeans clustering
def plot_clustered_data(X, labels, centers):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x', s=200, label='Centroids')
    plt.title('KMeans Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()
 
kmeans = KMeans(n_clusters=6,n_init=10, random_state=42)
means = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
# let's use wcss and elbow method to find the optimal number of clusters
def calculate_wcss(X, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss
wcss = calculate_wcss(X, max_clusters=10)
# Plot the WCSS to find the elbow point
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('WCSS vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid()
plt.show()
# Plot the clustered data
print(means)
plot_clustered_data(X, means, centers)