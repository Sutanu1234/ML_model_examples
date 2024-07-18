#Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Data Collection & Analysis
customer_data = pd.read_csv('Mall_Customers.csv')
customer_data.head()

# finding the number of rows and columns
customer_data.shape
# getting some informations about the dataset
customer_data.info()
# checking for missing values
customer_data.isnull().sum()

#Choosing the Annual Income Column & Spending Score column
X = customer_data.iloc[:,[3,4]].values


#Choosing the number of clusters WCSS -> Within Clusters Sum of Squares
# finding wcss value for different number of clusters
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', max_iter = 300, n_init = 10, random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)
  
# plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()  


#Optimum Number of Clusters = 5 Training the k-Means Clustering Model
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter = 300, n_init = 10, random_state=42)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)

#Visualizing all the Clusters
# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=100, c='green', label='Standard')
plt.scatter(X[Y==1,0], X[Y==1,1], s=100, c='red', label='Careful')
plt.scatter(X[Y==2,0], X[Y==2,1], s=100, c='yellow', label='Sensible')
plt.scatter(X[Y==3,0], X[Y==3,1], s=100, c='violet', label='Careless')
plt.scatter(X[Y==4,0], X[Y==4,1], s=100, c='blue', label='Target')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()







