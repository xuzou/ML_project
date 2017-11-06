import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class exampleSVM(object):
	def __init__(self,plotx=2,ploty=2):
		self._h=0.02
		self._C = 1.0
		self._degree = 3
		self._gamma = 0.7
		self._tol=1e-4
		self._maxIters=1000,
		self._fig = plt.figure()
		self._figNum = self._fig.number
		self._fig.subplots_adjust(wspace=0.4,hspace=0.4)
		self._subPlotNum = 1
		self._subPlotMax = 4
		self._plotx = plotx
		self._ploty = ploty

	def checkSubPlot(self):
		if self._subPlotNum <= self._subPlotMax:
			return True
		return False

	def Linear(self,X=None,y=None,tol=None,maxIters=None,C=None,h=None,replacePlot=None):
		title = "Linear"
		if replacePlot and replacePlot <= self._subPlotMax:
			plotNum = int(float("{1}{2}{0}".format(replacePlot,self._plotx,self._ploty)))
		elif not self.checkSubPlot():
			plotNum = int(float("{1}{2}{0}".format(1,self._plotx,self._ploty)))
			self._subPlotNum = 1
		kernel = 'linear'
		if tol is None:
			tol = self._tol
		if maxIters is None:
			maxIters = self._maxIters
		if C is None:
			C = self._C
		else: statStr = " C={}".format(C)
		if h is None:
			h = self._h
		else: statStr = "{}, step={}".format(h)
		if X is None or y is None:
			dSet = datasets.load_breast_cancer()
			X = dSet.data[:,:2]
			y = dSet.target
		# get svc
		linsvc = svm.SVC(kernel=kernel,tol=tol,max_iter=maxIters,C=C).fit(X,y)
		self.__plot__(linsvc,title,X,y)

	def LinLinear(self,X=None,y=None,tol=None,maxIters=None,C=None,h=None,replacePlot=None):
		title = "Linear+"
		if replacePlot and replacePlot <= self._subPlotMax:
			plotNum = int(float("{1}{2}{0}".format(replacePlot,self._plotx,self._ploty)))
		elif not self.checkSubPlot():
			plotNum = int(float("{1}{2}{0}".format(1,self._plotx,self._ploty)))
			self._subPlotNum = 1
		if tol is None:
			tol = self._tol
		if maxIters is None:
			maxIters = self._maxIters
		if C is None:
			C = self._C
		else: statStr = " C={}".format(C)
		if h is None:
			h = self._h
		else: statStr = "{}, step={}".format(h)
		if X is None or y is None:
			dSet = datasets.load_breast_cancer()
			X = dSet.data[:,:2]
			y = dSet.target
		# get svc
		linLinsvc = svm.LinearSVC(C=C).fit(X,y)
		self.__plot__(linLinsvc,title,X,y)

	def Poly(self,X=None,y=None,tol=None,maxIters=None,degree=None,C=None,h=None,replacePlot=None):
		title = "Poly"
		if replacePlot and replacePlot <= self._subPlotMax:
			plotNum = int(float("{1}{2}{0}".format(replacePlot,self._plotx,self._ploty)))
		elif not self.checkSubPlot():
			plotNum = int(float("{1}{2}{0}".format(1,self._plotx,self._ploty)))
			self._subPlotNum = 1
		kernel = 'poly'
		if tol is None:
			tol = self._tol
		if maxIters is None:
			maxIters = self._maxIters
		if C is None:
			C = self._C
		else: statStr = " C={}".format(C)
		if h is None:
			h = self._h
		else: statStr = "{}, step={}".format(h)
		if X is None or y is None:
			dSet = datasets.load_breast_cancer()
			X = dSet.data[:,:2]
			y = dSet.target
		# get svc
		polySvc = svm.SVC(kernel=kernel,C=C,tol=tol,max_iter=maxIters).fit(X,y)
		self.__plot__(polySvc,title,X,y)

	def RBF(self,X=None,y=None,tol=None,maxIters=None,degree=None,C=None,h=None,replacePlot=None):
		title = "RBF"
		if replacePlot and replacePlot <= self._subPlotMax:
			plotNum = int(float("{1}{2}{0}".format(replacePlot,self._plotx,self._ploty)))
		elif not self.checkSubPlot():
			plotNum = int(float("{1}{2}{0}".format(1,self._plotx,self._ploty)))
			self._subPlotNum = 1
		kernel = 'rbf'
		if tol is None:
			tol = self._tol
		if maxIters is None:
			maxIters = self._maxIters
		if C is None:
			C = self._C
		else: statStr = " C={}".format(C)
		if h is None:
			h = self._h
		else: statStr = "{}, step={}".format(h)
		if X is None or y is None:
			dSet = datasets.load_breast_cancer()
			X = dSet.data[:,:2]
			y = dSet.target
		# get svc
		rbfSvc = svm.SVC(kernel=kernel,C=C,tol=tol,max_iter=maxIters).fit(X,y)
		self.__plot__(rbfSvc,title,X,y)

	def __plot__(self,svcHand,title,X,y):
		# set up some plot stuff
		title="Support Vector Machine: {} kernel".format(title)
		x_min,x_max=X[:,0].min-1,X[:,1].max()+1
		y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
		xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

		# get our figure
		plt.figure(self._figNum)
		# add plot and show decision boundaries
		plt.add_subplot(plotNum)
		# make sure window is open
		plt.draw()
		plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Spectral)
		# plot linsvc contours
		for i,cl in enumerate((linsvc)):
			Z = cl.predict(np.c_[xx.ravel(),yy.ravel()])
		Z = Z.reshape(xx.shape)
		self._fig.contourf(xx,yy,Z)
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.xlim(xx.min(),xx.max())
		plt.ylim(yy.min(),yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.title(title)
		plt.show()








