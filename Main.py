import pandas as pd

df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.info())
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print(X.head())
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

print(y_kmeans[:10])
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.savefig("customer_segmentation.png")
plt.show()
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.savefig("customer_segmentation_center.png", dpi=300)
plt.show()