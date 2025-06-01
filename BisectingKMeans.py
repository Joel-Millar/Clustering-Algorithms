import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def load_dataset(file_path='dataset'):
    # Input: file path as input with default set to dataset file
    # Output: The dataset file converted into a data frame using pandas library
    # Reads the dataset file and saves it for use for clustering

    try:
        # Load the dataset
        df = pd.read_csv(file_path, sep=" ", header=None, engine="python")

        data = df.iloc[:, 1:].values

        data = np.array(data)

    except:
        data = []

    return data

# Computes Silhouette Coefficient for every object in the dataset with respect to the given clustering and distance matrix
def silhouetteCoefficient(dataset, clusters, distMatrix):
    # Input: dataset, lists of datapoints in each class and a distance matrix
    # Output: Silhouette score 
    # Computes single silhouette coefficient value
    N = len(dataset)
    
    silhouette = [0 for i in range(N)]
    a = [0 for i in range(N)]
    b = [float('inf') for i in range(N)]
    
    for (i, obj) in enumerate(dataset):
        for (cluster_id, cluster) in enumerate(clusters):
            clusterSize = len(cluster)
            if i in cluster:
                # compute a(obj)
                if clusterSize > 1:
                    a[i] = np.sum(distMatrix[i][cluster])/(clusterSize-1)
                else:
                    a[i] = 0
            else:
                # compute b(obj)
                tempb = np.sum(distMatrix[i][cluster])/clusterSize
                if tempb < b[i]: 
                    b[i] = tempb
                
    for i in range(N):
        silhouette[i] = 0 if a[i] == 0 else (b[i]-a[i])/np.max([a[i], b[i]])
    
    return silhouette

# Computes Silhouette Coefficient for the dataset with respect to the given clustering and distance matrix
def silhouette(dataset, clusters, distMatrix):
    # Input: dataset, lists of datapoints in each class and a distance matrix
    # Output: average silhouette score
    # Computes average silhouette coefficient value for measuring the effectivess of clustering
    return np.mean(silhouetteCoefficient(dataset, clusters, distMatrix))


def ComputeDistance(a, b):
    # Input: 2 numpy arrays
    # Output: float or numpy array
    # Calculates the euclidean distance between the inputs
    return np.linalg.norm(np.array(a) - np.array(b))

def initialSelection(k, data):
    # Input: k number of classes, datapoints
    # List of cluster representations
    # Selects k number of cluster representations randomly
    return random.sample(list(data), k)

def kmeans(k, data, plus=[]):
    # Input: number of classes, datapoints, maximum iterations if convergence is not achieved
    # Output: Dictionary of datapoints as keypoints and values as classification
    # Assigns classes with cluster representatives and then re-calculates and reclusters until convergence criteria or max iterations is met

    # Randomly select the centroids from dataset
    if not plus:
        centroids = initialSelection(k, data)
    else:
        centroids = plus

    flag = True
    iterations = 0
    max_iter = 10

    while flag:
        classes = {}

        # Assign each point to the nearest centroid
        for point in data:
            distances = [ComputeDistance(point, c) for c in centroids]
            min_index = np.argmin(distances)
            classes[tuple(point)] = min_index

        # Update centroids
        new_centroids = []
        for clas in range(k):
            cluster_points = [p for p in classes if classes[p] == clas]
            if cluster_points:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(random.choice(data))

        # Check for convergence
        if np.allclose(new_centroids, centroids, atol=1e-4) or iterations >= max_iter:
            break

        centroids = new_centroids
        iterations += 1

    return classes, centroids

def computeSumfSquare(cluster):
    # Input: all values in a given class
    # Output: Distance numpy array or float
    # Calculate maximum distance between any two points in cluster
    max_dis = 0
    
    for i in range(len(cluster)):
        for j in range(i+1, len(cluster)):
            dis = ComputeDistance(cluster[i], cluster[j])
            if dis > max_dis:
                max_dis = dis
                
    return max_dis

def bisect_kMeans(k, clusters, leaves=2):
    # Input: k number of classes, list of lists containing each datapoint in each class, number of leaves i.e. current classes
    # Output: Recursive call with updated clusters list and incremented leaves and base case is the list of clustered values sorted into lists by their class
    # Applied Bisect KMeans algorithm using recursion where KMeans is applied to each cluster splitting them until the desire number of clusters is reached
    if leaves == k:
        return clusters
    
    # Select cluster with maximum diameter to split
    max_dis = 0
    first_leaf = 0
    for i in range(len(clusters)):
        current_dis = computeSumfSquare(clusters[i])
        if current_dis > max_dis:
            max_dis = current_dis
            first_leaf = i
    
    # Split the selected cluster using k-means (k=2)
    clustered, _ = kmeans(2, clusters[first_leaf])
    
    # Remove the original cluster and add new subclusters
    clusters.pop(first_leaf)
    
    # Convert clustered dict to two separate clusters
    new_cluster1 = []
    new_cluster2 = []
    for point, cluster_id in clustered.items():
        if cluster_id == 0:
            new_cluster1.append(point)
        else:
            new_cluster2.append(point)
    
    clusters.append(new_cluster1)
    clusters.append(new_cluster2)
    
    return bisect_kMeans(k, clusters, leaves+1)

def convert_to_dict(data):
    # Input: list of datapoints
    # Output: dictionary where key is the datapoint and the value is the classification
    # Takes list of list and converts to dictionary where datapoints are keys and the index of list is value
    dict1 = {}

    for i in range(len(data)):
        for x in data[i]:
            dict1[x] = i

    return dict1

def prep_bisect(clusters):
    # Input: dictionary where key is the datapoint and the value is the classification
    # Output: list of list of clustered datapoints
    # datapoints of the same class are collected and then output together as a list of lists, helper fuction to use KMeans and bisect_KMeans functions together

    num_class = max(clusters.values()) - min(clusters.values()) + 1

    values = []
    for c in range(num_class):
        clus = [i for i in clusters.keys() if clusters[i] == c]
        values.append(clus)

    return values

def plot_silhouette():
    # Input: None
    # Output: None
    # plots and saves a graph of dataset being clustered for k values from 1 to 9

    k_values = np.arange(2, 10)
    data = load_dataset()

    sil_scores = []

    N = len(data)
    distMatrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            distMatrix[i][j] = ComputeDistance(data[i], data[j])

    for k in k_values:
        # Run kMeans
        ch, _ = kmeans(2, data)
        #c3 = bisect_kMeans(k, prep_bisect(ch), 2)
        c3 = bisect_kMeans(k, prep_bisect(ch))
        cluster_dict = convert_to_dict(c3)

        if len(cluster_dict) < 2:
            print('Not enough data samples...')
            return 0

        # Convert cluster dictionary into list of clusters
        clusters = [[] for _ in range(k)]
        for point, cluster_id in cluster_dict.items():
            index = np.where((data == point).all(axis=1))[0][0]  # Find index of the point
            clusters[cluster_id].append(index)

        # Compute silhouette score using the implemented function
        sil_score = silhouette(data, clusters, distMatrix)
        sil_scores.append(sil_score)

    # Plot silhouette scores
    plt.plot(k_values, sil_scores, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.savefig('Bisect_kMeans_plot.png')

    plt.show()


def main():
    # Runs code and sets seed
    
    np.random.seed(50)

    plot_silhouette()



main()
