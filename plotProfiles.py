import sys
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import numpy as np
import matplotlib.animation as anim
from exampleSVM import exampleSVM

class Dataset(object):
	def __init__(self,name,X,y,featureNames=None,xfeature=0,yfeature=1):
		self.name = name
		self.features = featureNames
		if X.shape[1] > 2:
			self.X = np.concatenate((X[:,xfeature],X[:,yfeature]),axis=1)
		self.y = y
		self.x_min,self.x_max = X[:,0].min()-1,X[:,0].max()+1
		self.y_min,self.y_max = X[:,1].min()-1,X[:,1].max()+1
		self.graphProfiles = {}
		self.graphProfileNameTypes = {}
		self._addProfileType = {
				"svm" : addSVM,
				"kmeans" : addKM,
				"hierarchical" : addHierClust,
				"em" : addEM
					};
		self._editProfileType = {
				"svm" : editSVM,
				"kmeans" : editKM,
				"hierarchical" : editHierClust,
				"em" : editEM
					};
		self._plotProfileType = {
				"svm" : plotSVM,
				"kmeans" : plotKM,
				"hierarchical" : plotHierClust,
				"em" : plotEM 
					};
		self.updateProfileName = None
		return self

	def addGraphProfile(self,gName,gType):
		""" Adds default profile to dataset
			The profile can then be edited from the Interface
			"""
		if gType in self._addProfileType:
			self.graphProfileNameTypes[gName] = gType
			self.updateProfileName = gName
			self._addProfileType[gType]

	def editGraphProfile(self,gName):
		self.updateProfileName = gName
		self._editProfileType[self.graphProfileNameTypes[gName]]
	def plotGraphProfile(self,profName):
		self.updateProfileName = profName
		self._plotProfileType[self.updateProfileName]

	def addSVM(self,gName):
		self.graphProfiles[self.updateProfileName] = SupportVectorProfile()
		self.editSVM()
	def editSVM(self,gName=None):
		if gName == None:
			gName = self.updateProfileName
		print ("Hi I'm in editSVM!")

	def plotSVM(self,gName=None):
		if gName == None:
			gName = self.updateProfileName
		profile = graphProfiles[profName]
		# create mesh:
		xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
		title = "SVM with {} fitting, profile '{}' on {}".format(profile.kernel,profName,self.name)

	def addKM(self,gName):
		# add instantiation here
		self.editKM()
	def editKM(self,gName=None): # todo
		if gName == None:
			gName = self.updateProfileName

	def addHierClust(self,gName):
		# add instantiation here
		self.editHierClust()
	def editHierClust(self,gName=None): # todo
		if gName == None:
			gName = self.updateProfileName

	def addEM(self,gName):
		# add instantiation here
		self.editEM()
	def editEM(self,gName=None): # todo
		if gName == None:
			gName = self.updateProfileName

	def graphStats(self):
		return "Not Implemented"
	def dispStats(self,gName):
		if gName not in self.graphProfiles:
			return "No profile with name '{}'".format(str(gName))
		return str(self.graph_Profiles[gName].stats())
		
class SupportVectorProfile(object):
	def __init__(self,X,y,Xname='x',yname='y',stepsize=0.02,svmtype='SVC',kernel='linear',penalty_norm='l2',loss='squared-hinge',dual=True,
		tol=1e-4,error_penalty=1.0,multi_class='ovr',fit_intercept=True,
		intercept_scaling=1,class_weight=None,verbose=0,random_state=None,
		max_iter=1000):
		self.help_str = {
				"error penalty" : "<float> penalty parameter of error term",
				"loss" : "<string> svm ='hinge' or ='squared-hinge'",
				"penalty norm" : "<string> "
				};
		self.Xname = Xname
		self.X = X
		self.yname = yname
		self.Y = y
		self._stepsize = stepsize
		self._svmtype = svmtype # SVC, linear
		self._kernel = kernel
		self._penalty_norm = penalty_norm
		self._loss = loss
		self._dual = dual
		self._tol = tol
		self._error_penalty = error_penalty
		self._multi_class = multi_class
		self._fit_intercept = fit_intercept
		self._intercept_scaling = intercept_scaling
		self._class_weight = class_weight
		self._verbose = verbose
		self._random_state = random_state
		self._max_iter = max_iter

	def stats(self):
		statString = "Support Vector Machine profile '{}'\n".format(gName)
		statString = "Features: {} w.r.t. %s\n".format(self.Xname,self.yname)
		statString = "{}subtype = {}\t\t".format(statString,str(self.svmtype))
		statString = "{}stepsize = {}\t\t".format(statString,str(self.stepsize))
		statString = "{}kernel = {}\n".format(statString,str(self.kernel))
		statString = "{}errorpenalty = {}\t\t".format(statString,str(self.error_penalty))
		statString = "{}maxiterations = {}\t\t".format(statString,str(self.max_iter))
		statString = "{}stoptolerance = {}\n".format(statString,str(self.tol))
		statString = "{}sampleweights = {}".format(statString,str(self.class_weight))
		return statString

	def getFit(self):
		if self._svmtype == "SVC":
			return svm.SVC(kernel=self._kernel,C=self._error_penalty,tol=self._tol,max_iter=self._max_iter).fit(self.X,self.y)

	@property
	def stepsize(self):
		return self._stepsize
	@stepsize.setter
	def stepsize(self,arg):
		self._stepsize = arg
	@property
	def kernel(self):
		return self._kernel
	@kernel.setter
	def kernel(self,arg):
		self._kernel = arg
	@property
	def svmtype(self):
		return self._svmtype
	@svmtype.setter
	def svmtype(self,arg):
		self._svmtype = arg
	@property
	def penalty_norm(self):
		return self._penalty_norm
	@penalty_norm.setter
	def penalty_norm(self,arg):
		self._penalty_norm = arg
	@property
	def loss(self):
		return self._loss
	@loss.setter
	def loss(self,arg):
		self._loss = arg
	@property
	def dual(self):
		return self._dual
	@dual.setter
	def dual(self,arg):
		self._dual = arg
	@property
	def tol(self):
		return self._tol
	@tol.setter
	def tol(self,arg):
		self._tol = arg
	@property
	def error_penalty(self):
		return self._error_penalty
	@error_penalty.setter
	def error_penalty(self,arg):
		self._error_penalty = arg
	@property
	def multi_class(self):
		return self._multi_class
	@multi_class.setter
	def multi_class(self,arg):
		self._multi_class = arg
	@property
	def fit_intercept(self):
		return self._fit_intercept
	@fit_intercept.setter
	def fit_intercept(self,arg):
		self._fit_intercept = arg
	@property
	def intercept_scaling(self):
		return self._intercept_scaling
	@intercept_scaling.setter
	def intercept_scaling(self,arg):
		self._intercept_scaling = arg
	@property
	def class_weight(self):
		if self._class_weight is not None:
			return "variable"
		return "unit"
	@class_weight.setter
	def class_weight(self,point,weight):
		""" 'point' as a one-indexed value
		"""
		if self._class_weight is None:
			self._class_weight = np.array([1]*len(self._dataset.X))
		self._class_weight[point-1] = weight
	@property
	def verbose(self):
		return self._verbose
	@verbose.setter
	def verbose(self,arg):
		self._verbose = arg
	@property
	def random_state(self):
		return self._random_state
	@random_state.setter
	def random_state(self,arg):
		self._random_state = arg
	@property
	def max_iter(self):
		return self._max_iter
	@max_iter.setter
	def max_iter(self,arg):
		self._max_iter = arg