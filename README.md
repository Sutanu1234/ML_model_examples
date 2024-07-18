# Customer Segmentation using K-Means Clustering

## Overview
This project demonstrates customer segmentation using the K-Means clustering algorithm. The dataset used contains information about customers from a mall, including their annual income and spending score. The aim is to group customers into distinct clusters based on their spending habits and income levels.

## Steps Involved

### 1. Setting Environment Variable
To avoid memory leak warnings on Windows with MKL, the environment variable `OMP_NUM_THREADS` is set to 1.

### 2. Importing Dependencies
Necessary libraries for data manipulation, visualization, and clustering are imported.

### 3. Data Collection & Analysis
- **Loading Dataset:** The dataset is loaded and the first few rows are inspected.
- **Dataset Information:** The shape, info, and missing values in the dataset are checked.
- **Selecting Features:** The Annual Income and Spending Score columns are selected for clustering.

### 4. Determining the Number of Clusters
- **Calculating WCSS (Within Clusters Sum of Squares):** WCSS values for different numbers of clusters are calculated.
- **Plotting the Elbow Graph:** An elbow graph is plotted to determine the optimum number of clusters.

### 5. Training the K-Means Clustering Model
- **Optimum Number of Clusters:** The model is trained using the optimum number of clusters determined from the elbow graph.
- **Cluster Labels:** Labels for each data point based on their cluster are generated.

### 6. Visualizing the Clusters
- **Plotting Clusters and Centroids:** All the clusters and their centroids are plotted to visualize the customer groups.

## Graphs

### 1. Elbow Point Graph
- **Description:** Shows the WCSS values for different numbers of clusters.
- **Purpose:** Helps determine the optimum number of clusters.
- **Image:** ![image](https://github.com/user-attachments/assets/675ed220-8221-412a-a06c-54476f33cd76)

### 2. Customer Groups Visualization
- **Description:** Displays the clusters and their centroids.
- **Purpose:** Helps visualize the distinct customer segments.
- **Image:** ![image](https://github.com/user-attachments/assets/234c12b3-a351-489f-8832-5417ef9c2d94)

## Conclusion
The K-Means clustering algorithm successfully groups customers into five distinct segments based on their annual income and spending score. This segmentation helps in identifying different customer behaviors, which can be useful for targeted marketing strategies.





