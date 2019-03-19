import numpy as np
import sqlite3
import folium
from folium import plugins
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def map_data(df, map):
    for index, row in df.iterrows():
        folium.Circle(location=([row['LATITUDE'], row['LONGITUDE']]),
                      radius=1,
                      color='orangered',
                      fill=False
                      ).add_to(map)


def map_firepoints(X, map, number_of_centroids):
    colours = ['red', 'blue', 'green', 'cyan',
               'purple', 'orange', 'magenta', 'grey', 'white']
    for centroid_id in range(0, number_of_centroids):
        for i, j in zip(X[y_kmeans == centroid_id, 0], X[y_kmeans == centroid_id, 1]):
            folium.Circle(location=(i, j),
                          radius=1,
                          color=colours[centroid_id],
                          fill=False
                          ).add_to(map)


def map_centroids(centroids, map):
    for i, coord in enumerate(centroids):
        folium.CircleMarker(location=(coord[0], coord[1]),
                            radius=20,
                            color='black',
                            popup='Cluster Centroid: {0}, {1}'.format(
                                coord[0], coord[1]),
                            fill=True,
                            fill_color='black',
                            ).add_to(map)


def map_coordinate(coord, map):
    folium.Marker(location=[coord[0], coord[1]],
                  popup='Latitude: {0}, Longitude: {1}'.format(coord[0], coord[1])).add_to(map)


def map_firestation(centroid, map):
    folium.CircleMarker(location=(centroid[0], centroid[1]),
                        radius=15,
                        color='yellow',
                        fill=True,
                        fill_color='red',
                        popup='This is the nearest firefighting station',
                        ).add_to(map)


def closest_node(node, nodes):
    """ Function to find the closest cluster from a coordinate """
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


# Getting the dataset
cnx = sqlite3.connect('FPA_FOD_20170508.sqlite')
df = pd.read_sql_query(
    "SELECT LATITUDE,LONGITUDE FROM fires where STATE='CA' and FIRE_YEAR>2009 and FIRE_SIZE_CLASS IN ('D', 'E', 'F', 'G')", cnx)
X = df.iloc[:, [0, 1]].values

# Using the elbow method to find the optimal number of clusters
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++')
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# Fitting Kmeans Clustering to the dataset
kmeans = KMeans(n_clusters=6, init='k-means++')
y_kmeans = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_[:, [0, 1]]

# Folium map 1: Fire points
m = folium.Map([37.5, -120], tiles='Stamen Terrain', zoom_start=6)
map_data(df, m)
m.save('firemap1.html')

# Folium map 2: Fire points and clusters
m2 = folium.Map([37.5, -120], tiles='Stamen Terrain', zoom_start=6)
map_firepoints(X, m2, 6)
map_centroids(centroids, m2)
m2.save('firemap2.html')

# Folium Map 3: Inputted coordinate and clusters
m3 = folium.Map([37.5, -120], tiles='Stamen Terrain', zoom_start=6)
map_centroids(centroids, m3)
fire_lat = input('Enter latitude of reported fire: ')
fire_lon = input('Enter latitude of reported fire: ')
fire_coord = [float(fire_lat), float(fire_lon)]
map_coordinate(fire_coord, m3)
m3.save('firemap3.html')

# Folium Map 4: Inputted coordinate and closest cluster
m4 = folium.Map([37.5, -120], tiles='Stamen Terrain', zoom_start=6)
map_centroids(centroids, m4)
map_coordinate(fire_coord, m4)
nearest_centroid = centroids[closest_node(fire_coord, centroids)]
map_firestation(nearest_centroid, m4)
m4.save('firemap4.html')
