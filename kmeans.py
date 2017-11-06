import matplotlib.pyplot as plt
import numpy as np
#from numpy import *
from matplotlib.animation import FuncAnimation
import scipy.io as sio

#points = []
mat = u'F:/study/NEU/Machine Learning/hw_04/dataset2.mat'
data = sio.loadmat(mat)
points = np.matrix(data['data'])

k = input("Input the cluster number")
k = int(k)
j = input("Input the number of Runs")
j = int(j)

points = np.array(points)
#print(points)
pause = False
#points = np.matrix(mat['data'])
#print(points)
#a = np.matrix(mat['data'])


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])


fig = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
centroids = initialize_centroids(points, k)


#def init():
#    return


count = 0


def animate(i):
    if not pause:
        global centroids
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
        global count
        count = count + 1
        
        #print("Iteration Number %d" % (count))
        ax.cla()
        tt = "Iteration {}".format(count)
        print(tt)
        ax.text(0.1, 3.0, tt, fontsize=18)
        ax.scatter(points[:, 0], points[:, 1], c=closest)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
        plt.title('Kmeans')
    return


def onClick(event):
    global pause
    pause ^= True


# if __name__ == '__main__':
fig.canvas.mpl_connect('button_press_event', onClick)
anim = FuncAnimation(fig, animate, frames=j, interval=1000, repeat=False)
plt.show()
