from scipy.cluster.hierarchy import *
import numpy as np
import cluster as cl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import *
from matplotlib.pyplot import *
import scipy.io as scp


def hierarchicalClustering(input_data=None, num_clusters=2, dist_type='euclidean', max_iterations=None, start_point_index=None, outlier_range=None):
    global merged_original_indices
    global X
    global fig
    global data
    global color_array

    merged_indices = []
    merged_original_indices = []
    color_array = []

    threshold = 10000000

    np.set_printoptions(precision=5, suppress=True)
    ignored_labels = []
    np.random.seed(1515)

    if input_data is not None:
        X = input_data
    else:
        #a = np.random.multivariate_normal([10,0], [[3,1],[1,4]], size=[100,])
        #b = np.random.multivariate_normal([0, 20], [[3,1],[1,4]], size=[50,])
        #X = np.concatenate((a,b),)
        mat = scp.loadmat(os.getcwd() + '/data/dataset2.mat')
        X = np.array(mat['data'])
        X = X[0:200]

    outlier_list = []
    # set up the initial list of data
    data = []
    if outlier_range is not None:
        min_allowed_x = outlier_range[0][0]
        max_allowed_x = outlier_range[0][1]
        min_allowed_y = outlier_range[1][0]
        max_allowed_y = outlier_range[1][1]
        for i in range(len(X)):
            if X[i][0] > min_allowed_x and X[i][0] < max_allowed_x and X[i][1] > min_allowed_y and X[i][1] < max_allowed_y:
                data.append(cl.Cluster([X[i]], originalLabels=[i]))
            else:
                np.delete(X,i)
                print(i)
                outlier_list.append(i)
    else:
        for i in range(len(X)):
            data.append(cl.Cluster([X[i]], originalLabels=[i]))

    max_x = np.max(X[:,0])
    min_x = np.min(X[:,0])
    max_y = np.max(X[:,1])
    min_y = np.min(X[:,1])
    # create distance matrix
    distances = np.zeros([len(X), len(X)])
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                distances[i][j] = 0
            else:
                if dist_type=='euclidean':
                    distances[i][j] = euclideanDistance(data[i].mean, data[j].mean)
                elif dist_type == 'max':
                    distances[i][j] = maxDistance(data[i], data[j])
                elif dist_type == 'manhattan':
                    distances[i][j] = manhattanDistance(data[i].mean, data[j].mean)

    # find lowest nonzero distance
    lowestDistance = np.min(distances[np.nonzero(distances)])
    iterations = 0        
    if max_iterations is None:
        num_iterations = 1000
    else:
        num_iterations = max_iterations

    while lowestDistance < threshold and len(data)-1 > num_clusters and iterations < num_iterations:
        # get the indices and combine those datapoints
        indices = np.where(distances == lowestDistance)
        index1 = indices[0][0]
        index2 = indices[0][1]
        if start_point_index is not None and iterations == 0:
            index1 = start_point_index[0]
            index2 = start_point_index[1]

        merged_indices += [[index1, index2]]

        # create a new class with those datapoints as the datapoints
        dataPoints = np.concatenate((data[index1].dataPoints, data[index2].dataPoints),)
        indices = np.concatenate((data[index1].originalLabels, data[index2].originalLabels),)
        merged_original_indices.append(indices)
        newInstance = cl.Cluster(dataPoints, indices)
        if index1 > index2:
            del data[index1]
            del data[index2]
        else:
            del data[index2]
            del data[index1]
        data.append(newInstance)
        # create new matrix
        distances = np.zeros([len(data), len(data)])
        for i in range(len(data)):
           for j in range(len(data)):
                if i == j:
                    distances[i][j] = 0
                else:
                    if dist_type == 'euclidean':
                        distances[i][j] = euclideanDistance(np.array(data[i].mean), np.array(data[j].mean))
                    elif dist_type == 'max':
                        distances[i][j] = maxDistance(data[i], data[j])
                    elif dist_type == 'manhattan':
                        distances[i][j] = manhattanDistance(data[i].mean, data[j].mean)

        lowestDistance = np.min(distances[np.nonzero(distances)])
        if max_iterations is not None:
            iterations += 1
        else:
            iterations = 1
    color_array = np.empty((len(merged_indices), len(X)), float)
    row_colors = np.random.random((len(X)))
    print(outlier_list)
    for i in range(len(merged_indices)):
        color_array[i] = row_colors
        for outlier in outlier_list:
            color_array[i][outlier] = 1.0
    for i in range(1,len(merged_indices)):
        for j in range(1,len(merged_original_indices[i])):
            color_array[i][merged_original_indices[i][j]] = color_array[i][merged_original_indices[i][0]]
        for k in range(i, len(merged_indices)):
            color_array[k] = color_array[i]
    numframes = len(merged_indices)
    fig, ax = subplots()
    ax.set_xlim([min_x - 1, max_x + 1])
    ax.set_ylim([min_y-1, max_y+1])
    anim = animation.FuncAnimation(fig, animate, frames = range(numframes), interval=100, init_func=init, blit=True, repeat=0)
    show()

def init():
    return []
    
def animate(i):
    newpoints = ()
    for k in range(len(X)):
        val = matplotlib.colors.rgb2hex(cm.Paired(color_array[i][k]))
        if k in merged_original_indices[i]:
            newpoints += (X[k,0], X[k,1], val)
        else:
            newpoints += (X[k,0], X[k,1], val)
    animlist = plot(*newpoints, marker='o', markersize=8)
    return animlist
    

def euclideanDistance(data1, data2):
    numDims = np.size(data1)
    distance = 0
    for i in range(numDims):
        distance += (data1[i] - data2[i])**2
    distance = np.sqrt(distance)
    return distance

def maxDistance(cluster1, cluster2):
    maxDistance = 0
    for i in range(len(cluster1.dataPoints)):
        for j in range(len(cluster2.dataPoints)):
            distance = euclideanDistance(cluster1.dataPoints[i], cluster2.dataPoints[j])
            if distance > maxDistance:
                maxDistance = distance
    return maxDistance

def manhattanDistance(data1, data2):
    numDims = np.size(data1)
    distance = 0 
    for i in range(numDims):
        distance += np.abs(data1[i] - data2[i])
    return distance
