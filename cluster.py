import numpy as np

class Cluster:
    mean = 0
    dataPoints = []
    color = 'r'

    def __init__(self, dataPoints=[], originalLabels=[]):
        self.dataPoints = dataPoints
        self.originalLabels = originalLabels
        if len(dataPoints) > 0:
            self.numDims = np.size(dataPoints[0])
            mean = np.array(dataPoints[0])
            if (len(dataPoints) > 1):
                for i in range(1, len(dataPoints)):
                    mean += dataPoints[i]
            mean /= len(dataPoints)
            self.mean = np.squeeze(mean)
        else:
            self.mean = 0
            self.numDims = 0
