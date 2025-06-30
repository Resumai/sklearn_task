import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def plot_clustered_data(X, labels, centers):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x', s=200, label='Centroids')
    plt.title('KMeans Clustering Results')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_wcss(X, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss


def plot_wcss(wcss):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('WCSS vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, 11))
    plt.grid()
    plt.show()


df = pd.read_csv("mall_customers/Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

model = KMeans(n_clusters=5, n_init=10, random_state=42)
x_pred = model.fit_predict(X)
centers = model.cluster_centers_
df['Cluster'] = x_pred

wcss = calculate_wcss(X, max_clusters=10)

# plot_clustered_data(X, x_pred, centers)
# plot_wcss(wcss)

# Calculate cluster summary statistics
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_counts = df['Cluster'].value_counts()

# Prepare insights
insights = []
for cluster_id, row in cluster_summary.iterrows():
    count = cluster_counts[cluster_id]
    avg_income = row['Annual Income (k$)']
    avg_score = row['Spending Score (1-100)']
    insights.append(f"Cluster {cluster_id}: {count} customers, avg income ~{avg_income:.1f}, avg score ~{avg_score:.1f}")

for insight in insights:
    print(insight)
