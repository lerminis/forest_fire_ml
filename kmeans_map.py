import numpy as np
import sqlite3
import folium
from folium import plugins
import pandas as pd
from sklearn.cluster import KMeans


def map_firepoints(df, map):
    for index, row in df.iterrows():
        folium.Circle(location=([row['LATITUDE'], row['LONGITUDE']]),
                      radius=1,
                      color='orangered',
                      fill=False
                      ).add_to(map)


def map_centroids(centroids, map):
    for i, coord in enumerate(centroids):
        folium.CircleMarker(location=(coord[0], coord[1]),
                            radius=12,
                            color='blue',
                            popup='Cluster Centroid: {0}, {1}'.format(
                                coord[0], coord[1]),
                            fill=True,
                            fill_color='blue',
                            ).add_to(map)


def map_coordinate(coord, map):
    folium.Marker(location=[coord[0], coord[1]],
                  popup='Latitude: {0}, Longitude: {1}'.format(coord[0], coord[1])).add_to(map)


def map_firestation(centroid, map):
    folium.CircleMarker(location=(centroid[0], centroid[1]),
                        radius=15,
                        color='red',
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

# Fitting Kmeans Clustering to the dataset
kmeans = KMeans(n_clusters=12, init='k-means++')
y_kmeans = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_[:, [0, 1]]

# Folium map 1: Fire points
m1 = folium.Map([37.5, -120], tiles='Stamen Terrain', zoom_start=6)
map_firepoints(df, m1)
m1.save('firemap1.html')

# Folium map 2: Fire points and clusters
m2 = folium.Map([37.5, -120], tiles='Stamen Terrain', zoom_start=6)
map_firepoints(df, m2)
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
