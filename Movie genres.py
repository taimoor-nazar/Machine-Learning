#Movie Genre Classification using K means

from math import sqrt
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator

#getting ratings data
userColumns = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep = '\t', names = userColumns)

#getiing movie genres
movieColumns = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('u.item', sep='|', names=movieColumns, encoding = 'latin-1')

#calculating avg Ratings
avgRatings = ratings.groupby('movieId')['rating'].mean().reset_index()
avgRatings.columns = ['movieId', 'avgRating']

movieData = pd.merge(avgRatings, movies[['movieId'] + movieColumns[5:]], on='movieId')

trainingFeatures = movieData.drop('movieId', axis=1)

#scaling the data
scaler = MinMaxScaler()
scaledFeatures = scaler.fit_transform(trainingFeatures)

#finding optimal number of clusters
sse = []
k_range = int(sqrt(1682/2)) #using heuristic function to get the range of k
#print(k_range)

for k in range(1, k_range):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaledFeatures)
    sse.append(km.inertia_)

#using built in elbow finder
elbowLocator = KneeLocator(range(1, k_range), sse, curve='convex', direction='decreasing')
elbow = elbowLocator.elbow
#print(elbow)

#Clustering
model = KMeans(n_clusters=elbow, random_state=42)
prediction = model.fit_predict(scaledFeatures)

movieData['cluster'] = prediction
centers = model.cluster_centers_

#visualizing the clusters and centers

#using PCA to transform multi dimensional features into 2D(this is not perfect hence why some points in different clusters appear to overlap)
pca = PCA(n_components=2)
newFeatures = pca.fit_transform(scaledFeatures)
scaledCenters = pca.transform(centers)
#print(scaledCenters)

df = pd.DataFrame(newFeatures, columns=['C1', 'C2'])
df['cluster'] = prediction

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='C1', y='C2', hue='cluster', palette='tab10', s=60)
plt.scatter(scaledCenters[:, 0], scaledCenters[:, 1], marker='*', label='Centers', color = 'black')

for i, (x, y) in enumerate(scaledCenters):
    plt.text(x, y, f'Cluster {i}', fontsize=10, ha='center', va='bottom', weight='bold')

plt.title('KMeans Clusters Visualized')
plt.xlabel('Genres Combined')
plt.ylabel('Ratings')
plt.legend(title='Cluster')
plt.legend()
plt.show()

#using original centers to understand what each cluster represents
originalCenters = scaler.inverse_transform(centers)
center_df = pd.DataFrame(originalCenters, columns=trainingFeatures.columns)
#print(center_df)

plt.figure(figsize=(12, 6))
sns.heatmap(center_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Cluster Centers Representation")
plt.xlabel("Genres")
plt.ylabel("Cluster")
plt.show()