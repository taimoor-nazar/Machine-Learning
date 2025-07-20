import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
#print(df)

#plotting the data

plt.title('Data')
plt.xlabel('Income($)')
plt.ylabel('Spending Score(1-100)')
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.show()

#From the graph we can predict that optimal number of clusters may be 5 clusters
#Create the model and train it to find which cluster each points belong to

model = KMeans(n_clusters=5, random_state=42)
model.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
prediction = model.predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

#adding the cluster prediction to the dataframe

df['cluster'] = prediction

#visualizing the clusters

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

plt.title('Clusters')
plt.xlabel('Income($)')
plt.ylabel("Spending Score(1-100)")
plt.scatter(df1['Annual Income (k$)'], df1['Spending Score (1-100)'], color = 'red', label = 'Cluster 1')
plt.scatter(df2['Annual Income (k$)'], df2['Spending Score (1-100)'], color = 'green', label = 'Cluster 2')
plt.scatter(df3['Annual Income (k$)'], df3['Spending Score (1-100)'], color = 'yellow', label = 'Cluster 3')
plt.scatter(df4['Annual Income (k$)'], df4['Spending Score (1-100)'], color = 'blue', label = 'Cluster 4')
plt.scatter(df5['Annual Income (k$)'], df5['Spending Score (1-100)'], color = 'orange', label = 'Cluster 5')

#adding the centroids

centers = model.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], color = 'black', marker='*', label= 'Centroids')

plt.legend()
plt.show()

#In Kmeans elbow method is used to find the optimal number of clusters

sse = []
for k in range(1, 10):

  km = KMeans(n_clusters=k, random_state=42)
  km.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
  sse.append(km.inertia_) # inertia gives us the sum squared error for every number of clusters

#plot K, SSE graph and try to locate the elbow
k_range = range(1,10)
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_range, sse)
plt.show()

#I can see in this plot that our elbow forms at k = 5
#Hence my initial assumption was right

elbow = 5
plt.title('Elbow marked')
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_range, sse)
plt.scatter(elbow, sse[elbow - 1], color = 'black', label = 'Elbow(optimal number of clusters)')
plt.legend()
plt.show()
