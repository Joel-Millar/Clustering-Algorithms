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

def computeClusterRepresentatives(k, classes, data):
    # Input: k number of classes, classes of each datapoint, datapoints
    # Output: list of new cluster representation points
    # Calculates the mean of each class and makes that the new centroid
    new_centroids = []
    for clas in range(k):
        cluster_points = [p for p in classes if classes[p] == clas]
        if cluster_points:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(random.choice(data))
    
    return new_centroids

def assignClusterIds(centroids, data):
    # Input: Cluster representations, datapoints
    # Output: Dictionary where datapoints are keys and classes are the value
    # Assigns each datapoint to a class in a dictionary
    classes = {}

    for point in data:
        distances = [ComputeDistance(point, c) for c in centroids]
        min_index = np.argmin(distances)
        classes[tuple(point)] = min_index
    
    return classes

def kMeans(k, data, plus=[], max_iter=1000):
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

    while flag:
        classes = {}

        # Assign each point to the nearest centroid
        classes = assignClusterIds(centroids, data)

        # Update centroids
        new_centroids = computeClusterRepresentatives(k, classes, data)

        # Check for convergence
        if np.allclose(new_centroids, centroids, atol=1e-4) or iterations >= max_iter:
            break

        centroids = new_centroids
        iterations += 1

    return classes, centroids

def initialSelection(k, data):
    # Input: k number of classes, datapoints
    # List of cluster representations
    # Selects k number of cluster representations using kmeans++ algorithm
    
    data = np.array(data)  # Convert input data to NumPy array
    f_sp = []

    # Select first centroid randomly
    ip = np.random.randint(0, len(data))
    f_sp.append(data[ip].tolist())

    for _ in range(k - 1):
        dists = []

        # Compute distance from each point to the nearest centroid
        for point in data:
            nearest = min(ComputeDistance(point, c) for c in f_sp)
            dists.append(nearest)

        # Compute probabilities for next centroid selection
        dists = np.array(dists) ** 2  # Squaring distances for weighted probability
        probs = dists / dists.sum()  # Normalize probabilities

        # Select the next centroid based on probabilities
        next_cent = np.random.choice(len(data), p=probs)
        f_sp.append(data[next_cent].tolist())
    
    return f_sp

def kMeansplusplus(k, data):
    # Input: k number of clusters, datapoints
    # Output: Dictionary of clustered points as key and classification as value, list of cluster representatives
    # Combines the KMeans ++ cluster representative initialisation and then applies KMeans

    f_sp = initialSelection(k, data)
    # Run K-Means with selected centroids
    km, centroids = kMeans(k, data, f_sp)

    return km, centroids


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
        cluster_dict, centroids = kMeansplusplus(k, data)

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
    plt.savefig('kMeansplusplus_plot.png')

    plt.show()    

def main():
    # Runs code and sets seed
    np.random.seed(50)

    plot_silhouette()
main()
