#!/usr/local/bin/
import sys
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import numpy as np
import matplotlib.animation as anim
from exampleSVM import exampleSVM
from plotProfiles import Dataset, SupportVectorProfile

class UserIn:
	@classmethod
	def getInputs(self,*args):
		choices = list()
		for arg in args:
			query = "Enter value for {}: ".format(arg)
			choice = input(query)
			if not choice: 
				choices.append(None)
			else: 
				append(choice)
		return choices

	@classmethod
	def getInputByNumber(self,*args):
		""" returns index of choice
		"""
		prompt="Input Number: "
		args = list(args)
		ind = 1
		for arg in args:
			print("{}. {}".format(ind,arg))
			ind=ind+1
		return int(float(input(prompt)))

class ExamplePrograms(object):
	def __init__(self,dataset=None,X=None,Y=None):
		self.dataset=dataset
		if dataset:
			self.dataX = dataset.data[:,:2]
			self.dataY = dataset.target
		else:
			self.dataX = X
			self.dataY = Y
		self._SVM = exampleSVM()

	def emProgram(self):
		pass

	def kmProgram(self):
		pass

	def svmProgram(self):
		choice = UserIn.getInputByNumber("Linear","Linear+","Poly","RBF")
		while choice not in range(1,5):
			sys.stdout("Try again")
			choice = UserIn.getInputByNumber("Linear","Linear+","Poly","RBF")
		if choice is 1: self._SVM.Linear(X=self.dataX,y=self.dataY);
		elif choice is 2: self._SVM.LinLinear(X=self.dataX,y=self.dataY);
		elif choice is 3: self._SVM.Poly(X=self.dataX,y=self.dataY);
		elif choice is 4: self._SVM.RBF(X=self.dataX,y=self.dataY)
		else: print("Error")

	def hcProgram(self):
		pass

class Interface():
	def __init__(self,data1=None,data2=None,name1=None,name2=None):
		self._MAIN = 2
		self._ADDDATA = 3
		self._LISTDATA = 4
		self._GRAPHING = 7
		self._EXIT = 0
		self._PREV = 1
		self._ADDDEFAULT = 5
		self._ADDFILE = 6
		self._GRAPHPROFILE = 7
		self._DATASETUP = 8
		self._GRAPHING = 9
		self._GENDATA = 10
		self._GRAPHNEW = 11
		self._GRAPHEDIT = 12
		self._GRAPHSTATS = 13
		self._DRAWGRAPH = 14
		self._DRAWANIM = 15
		self._DRAWCOMPARE = 16
		self._PROGRAM = 17
		self._currentMenu = None
		self._prevMenu = None
		self._datasets = {} # name : Dataset
		self._currentSet = None
		self._defaultSets = {}
		self._em = "em"
		self._km = "km"
		self._svm = "svm"
		self._hc = "hc"
		self._exProgs = None
		if data1:
			self._defaultSets[name1]=data1
		if data2:
			self._defaultSets[name2]=data2
			self._exProgs = ExamplePrograms(dataset=data2)

		self._MENUTITLES = {
				self._EXIT : ">>> EXIT",
				self._PROGRAM: ">>> RUN PROGRAM",
				self._PREV : ">>> ",
				self._MAIN : ">>> MAIN",
				self._ADDDATA : ">>> ADD DATA",
				self._LISTDATA : ">>> LIST DATASETS",
				self._DATASETUP : ">>> SETUP DATASET",
				self._GRAPHPROFILE : ">>> CLASSIFIERS",
				self._ADDDEFAULT : ">>> ADD DEFAULT DATASET",
				self._ADDFILE : ">>> ADD DATASET FROM FILE",
				self._GENDATA : ">>> GENERATE DATA",
				self._GRAPHNEW : ">>> CREATE NEW CLASSIFIER",
				self._GRAPHEDIT : ">>> EDIT CLASSIFIER",
				self._GRAPHSTATS : ">>> PROFILE STATS",
				self._DRAWGRAPH : ">>> PLOT PROFILE",
				self._DRAWANIM : ">>> PLOT ANIMATION",
				self._DRAWCOMPARE : ">>> PLOT STATIC",
				"def" : ">>> err"
					}

		self._MENUPROMPTS = {
				self._EXIT : "Exit",
				self._PROGRAM : "Run Example Program",
				self._PREV : "Return to Previous Menu",
				self._ADDDATA : "Add dataset",
				self._LISTDATA : "Datasets- Details and Graphing",
				self._ADDDEFAULT : "Add from available defaults",
				self._ADDFILE : "Add from file",
				self._GRAPHSTATS : "Current profile stats",
				self._GRAPHPROFILE : "Edit or set up new graph profile",
				self._GRAPHNEW : "New profile",
				self._GRAPHEDIT : "Edit Current Profile",
				self._DRAWGRAPH : "Plot Graph",
				self._DRAWANIM : "Plot Animated Graph",
				self._DRAWCOMPARE : "Plot static graph(s)",
				"def" : "err"
					}

		self._ClassificationPrompt = {
				self._em : "Expectation-Maximization",
				self._km : "K-Means",
				self._svm : "Support Vector Machine",
				self._hc : "Hierarchical Clustering"
					}

	def _GOTOMENU(self,menu=None):
		print("\n")
		if menu is self._EXIT : 
			print("{}".format(self._MENUTITLES[self._EXIT]))
			self.exitMenu()
		elif menu is self._PREV : 
			print("{}".format(self._MENUTITLES[self._PREV]))
			self.prevMenu()
		elif menu is self._MAIN : 
			print("{}".format(self._MENUTITLES[self._MAIN]))
			self.mainMenu()
		elif menu is self._ADDDATA : 
			print("{}".format(self._MENUTITLES[self._ADDDATA]))
			self.addMenu()
		elif menu is self._PROGRAM :
			print("{}".format(self._MENUTITLES[self._PROGRAM]))
			self.progMenu()
		elif menu is self._LISTDATA : 
			print("{}".format(self._MENUTITLES[self._LISTDATA]))
			self.dataMenu()
		elif menu is self._DATASETUP : 
			print("{}".format(self._MENUTITLES[self._DATASETUP]))
			self.profileMenu()
		elif menu is self._GRAPHPROFILE : 
			print("{}".format(self._MENUTITLES[self._GRAPHPROFILE]))
			self.graphProfile()
		elif menu is self._ADDDEFAULT : 
			print("{}".format(self._MENUTITLES[self._ADDDEFAULT]))
			self.addDefaultDataMenu()
		elif menu is self._ADDFILE : 
			print("{}".format(self._MENUTITLES[self._ADDFILE]))
			self.addFileDataMenu()
		elif menu is self._GENDATA : 
			print("{}".format(self._MENUTITLES[self._GENDATA]))
			self.generateDataSamples()
		elif menu is self._GRAPHNEW : 
			print("{}".format(self._MENUTITLES[self._GRAPHNEW]))
			self.setUpGraphProfile()
		elif menu is self._GRAPHEDIT : 
			print("{}".format(self._MENUTITLES[self._GRAPHEDIT]))
			self.editGraphProfile()
		elif menu is self._GRAPHSTATS : 
			print("{}".format(self._MENUTITLES[self._GRAPHSTATS]))
			self.graphProfileStats()
		elif menu is self._DRAWGRAPH : 
			print("{}".format(self._MENUTITLES[self._DRAWGRAPH]))
			self.drawGraph()
		elif menu is self._DRAWANIM : 
			print("{}".format(self._MENUTITLES[self._DRAWANIM]))
			self.drawAnimated()
		elif menu is self._DRAWCOMPARE : 
			print("{}".format(self._MENUTITLES[self._DRAWCOMPARE]))
			self.drawStatic()
		else: self.invalidMenu()

	def startUp(self):
		self._GOTOMENU(menu=self._MAIN)

	def printMenuOptions(self,*args):
		ind = 0
		args = list(args) + [self._EXIT]
		#print("args: {}".format(args))
		for arg in args:
			print ("{}. {}".format(ind,self._MENUPROMPTS.get(arg,"def")))
			ind = ind + 1
		choice = int(float(input("\nInput number for choice: ")))
		#print("{} -> {}".format(choice,args[choice]))
		self._GOTOMENU(menu=args[choice])

	def mainMenu(self):
		self._prevMenu = self._MAIN
		self._currentMenu = self._MAIN
		# print menu
		# self.printMenuOptions(self._ADDDATA,self._LISTDATA,self._DRAWGRAPH,self._PROGRAM)
		self.printMenuOptions(self._PROGRAM)

	def progMenu(self):
		self._prevMenu = self._MAIN
		self._currentMenu = self._PROGRAM
		for key,val in self._ClassificationPrompt.items():
			print ("{}. {}".format(key,val))
		choice = input("\nInput short name for choice or 'cancel' to return: ")
		while choice not in self._ClassificationPrompt:
			choice = input("\nTry again, Input short name for choice or 'cancel to return': ")
		if choice == self._em:
			self._exProgs.emProgram()
		elif choice == self._km:
			self._exProgs.kmProgram()
		elif choice == self._svm:
			self._exProgs.svmProgram()
		elif choice == self._hc:
			self._exProgs.hcProgram()
		else:
			self._GOTOMENU(self._prevMenu)
		self._GOTOMENU(self._prevMenu)

	def _runProg(self,progtype):
		print ("Running {} program...".format(self._ClassificationPrompt[progtype]))
		if progtype == "em": 
			ExamplePrograms.emProgram()
		elif progtype == "kmeans" : 
			ExamplePrograms.kmProgram()
		elif progtype == "hierarchical" :
			ExamplePrograms.hcProgram()
		else: ExamplePrograms.svmProgram()

	def addMenu(self):
		self._prevMenu = self._MAIN
		self._currentMenu = self._ADDDATA
		self.printMenuOptions(self._PREV,self._ADDDEFAULT,self._ADDFILE)

	def addDefaultDataMenu(self):
		self._prevMenu = self._ADDDATA
		self._currentMenu = self._ADDDEFAULT
		ind = 0
		print("Available datasets:")
		for key in self._defaultSets:
			print("   {}".format(key))
		choice = input("Enter name of dataset, or 'cancel' to return: ")
		while choice not in self._defaultSets:
			choice = input("Try again, Enter name of dataset, or 'cancel' to return: ")
		if choice == "cancel":
			self._GOTOMENU(menu=self._prevMenu)
		self._DefaultData(str(choice))
	
	def _DefaultData(self,setName):
		try:
			temp = self._defaultSets[setName]
			print ("Dataset '{}' loaded".format(setName))
		except:
			print ("Dataset load failed")
		try:
			features = temp.target_names.flatten().tolist()
			print ("Features: {}".format(features))
			ind = 1
			for val in temp.target_names.flatten().tolist():
				print ("   {}. {}".format(ind,val))
				ind = ind+1
			xInd = int(float(input("Number of x axis feature: ")))
			while xInd not in temp.target:
				xInd = int(float(input("Try again, Number of x feature: ")))
			yInd = int(float(input("Number of y axis feature: ")))
			while not (yInd in temp.target and yInd != xInd):
				yInd = int(float(input("Try again, Number of y feature: ")))
			X = np.concatenate(temp.data[:,xInd],temp.data[:,yInd])
			featNames = [temp.target_names(xInd),temp.target_names(yInd)]
			self._datasets[setName] = Dataset(name=setName,X=X,y=temp.target,featureNames=featNames)
			print ("Dataset '{}' loaded".format(setName))
		except:
			print ("Dataset setup failed")
			sys.exit()
		self._GOTOMENU(menu=self._prevMenu)

	def addFileDataMenu(self):
		self._prevMenu = self._ADDDATA
		self._currentMenu = self._ADDFILE
		# todo
		print ("Not implemented")
		self._GOTOMENU(menu=self._prevMenu)

	def generateDataSamples(self):
		self._prevMenu = self._ADDDATA
		self._currentMenu = self._GENDATA
		# todo
		print ("Not implemented")
		self._GOTOMENU(menu=self._prevMenu)

	def dataMenu(self):
		self._prevMenu = self._MAIN
		self._currentMenu = self._LISTDATA
		if not self._datasets:
			print ("No datasets loaded")
			self._GOTOMENU(menu=self._prevMenu)
		print ("Loaded Datasets:")
		for key in self._datasets:
			print (" - {}".format(key))
		choice = 0
		while (choice != 'cancel' or choice not in self._datasets):
			choice = input("Enter name of dataset to work with or 'cancel' to return: ")
		if choice == "cancel":
			self._GOTOMENU(menu=self._prevMenu)
		# set current dataset
		self._currentSet = self._datasets[choice]
		self._GOTOMENU(menu=self._DATASETUP)

	def profileMenu(self):
		self._prevMenu = self._LISTDATA
		self._currentMenu = self._DATASETUP
		# print stats
		print("NAME: {}".format(self._currentSet.name))
		print(" - Features: {}".format(str(self._currentSet.features)))
		print(" - Num samples: {}".format(self._currentSet.x_max))
		print(" - Num features: {}".format(self._currentSet.y_max))
		print("Graph profiles: {}".format(str(self._currentSet.graph_Profiles)))
		self.printMenuOptions(self._PREV,self._GRAPHSTATS,self._GRAPHPROFILE)

	def graphProfileStats(self):
		self._prevMenu = self._DATASETUP
		self._currentMenu = self._GRAPHSTATS
		print ("{}".format(self._currentSet.graphStats()))
		self._GOTOMENU(menu=self._prevMenu)

	def graphProfile(self):
		self._prevMenu = self._DATASETUP
		self._currentMenu = self._GRAPHPROFILE
		self.printMenuOptions(self._PREV,self._GRAPHNEW,self._GRAPHEDIT)

	def setUpGraphProfile(self):
		self._prevMenu = self._GRAPHPROFILE
		self._currentMenu = self._GRAPHNEW
		# select type of classifier
		print ("Classification types: ")
		for key, value in self._ClassificationPrompt.items():
			print("For new {} classification profile, type '{}'".format(value,key))
		gType = input("classifier? ")
		while gType not in self._ClassificationPrompt.items():
			gType = input("Invalid choice. Classification? ")
		# get name of profile
		gName = input("Name new profile or type 'cancel' to return: ")
		if gName == "cancel":
			self._GOTOMENU(menu=self._prevMenu)
		self._currentSet.addGraphProfile(gName,gType)
		self._GOTOMENU(menu=self._prevMenu)

	def editEM(self):
		self._prevMenu = self._DATASETUP
		self._currentMenu = self._GRAPHPROFILE
		# perform setup
		# return to datasetup
		print ("Not Implemented")
		self._GOTOMENU(menu=self._prevMenu)

	def editKM(self):
		self._prevMenu = self._DATASETUP
		self._currentMenu = self._GRAPHPROFILE
		# perform setup
		# return to datasetup
		print ("Not Implemented")
		self._GOTOMENU(menu=self._prevMenu)

	def editSVM(self):
		self._prevMenu = self._DATASETUP
		self._currentMenu = self._GRAPHPROFILE
		# perform setup
		# return to datasetup
		print ("Not Implemented")
		self._GOTOMENU(menu=self._prevMenu)

	def editHC(self):
		self._prevMenu = self._DATASETUP
		self._currentMenu = self._GRAPHPROFILE
		# perform setup
		# return to datasetup
		print ("Not Implemented")
		self._GOTOMENU(menu=self._prevMenu)

	def editGraphProfile(self):
		self._prevMenu = self._GRAPHPROFILE
		self._currentMenu = self._GRAPHEDIT
		gName = None
		while gName != 'cancel' or gName not in self._currentSet.graphProfiles:
			gName = input("Name of profile to edit or 'cancel' to return: ")
		if gName == "cancel":
			self._GOTOMENU(menu=self._prevMenu)
		self._currentSet.editGraphProfile(gName)
		self._GOTOMENU(menu=self._prevMenu)

	def drawGraph(self):
		self._prevMenu = self._MAIN
		self._currentMenu = self._DRAWGRAPH
		# animate or compare?
		if not self._datasets:
			print ("No data to graph!")
			self._GOTOMENU(menu=self._prevMenu)
		self.printMenuOptions(self._PREV,self._DRAWCOMPARE,self._DRAWANIM)

	def drawStatic(self):
		print("Not Implemented")
		self._GOTOMENU(menu=self._prevMenu)

		self._prevMenu = self._DRAWGRAPH
		self._currentMenu = self._DRAWCOMPARE
		# choose profiles to graph
		self.listAll()
		print("Select up to four classifier profiles to plot")
		addSet = 1
		draw = {}
		while addSet != "no":
			dataName = input("Dataset name or 'cancel' to return: ")
			while dataName != 'cancel' and dataName not in self._datasets:
				dataName = input("Invalid selection, retry: ")
			if dataName == 'cancel':
				break
			profName = input("Profile name or 'cancel' to return: ")
			while profName != 'cancel' and profName not in self._datasets[dataName].graphProfiles:
				profName = input("Invalid selection, please retry: ")
			if profName == 'cancel':
				break
			if dataName not in draw:
				draw[dataName] = list([profName])
			else:
				draw[dataName].append(profName)
			if len(draw) > 4:
				break
			addSet = input("Add another profile? (yes/no): ")
		if not draw:
			self._GOTOMENU(menu=self._prevMenu)
		# perform drawing here!!!!
		print("Plotting...")
		# titles = list()
		# for setName,profList in draw:
		# 	for profName in profList:
		# 		# specify svm type and fit data
		# 		thisTitle = "%s profile '%s' of '%s' data" %(self._datasets[setName].graphProfileNameTypes[profName],profName,setName)
		# 		titles.append(thisTitle)

				

		# 		plt.subplot(1,1,1)
		# 		plt.subplots_adjust(wspace=0.4,hspace=0.4)

		# 		Z = svc.predict(np.c_[xx.ravel(),yy.ravel()])
		# 		Z = Z.reshape(xx.shape)
		# 		plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.8)

		# 		#plot
		# 		plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.coolwarm)
		# 		plt.xlabel('Sepal length')
		# 		plt.ylabel('Sepal width')
		# 		plt.xlim(xx.min(),xx.max())
		# 		plt.ylim(yy.min(),yy.max())
		# 		plt.xticks(())
		# 		plt.yticks(())
		# 		plt.title(titles[0])

		# 		plt.show()
		self._GOTOMENU(menu=self._prevMenu)

	def drawAnimated(self):
		print("Not Implemented")
		self._GOTOMENU(menu=self._prevMenu)

		self._prevMenu = self._DRAWGRAPH
		self._currentMenu = self._DRAWANIM
		self.listAll()
		dataName = None
		profName = None

		dataName = input("Dataset name or 'cancel' to return: ")
		while not (dataName == 'cancel' or dataName in self._datasets):
			dataName = input("Invalid selection, retry: ")
		if dataName == 'cancel':
			self._GOTOMENU(menu=self._prevMenu)
		profName = input("Profile name or 'cancel' to return: ")
		while not (profName == 'cancel' or profName in self._datasets[dataName].graphProfiles):
			profName = input("Invalid selection, please retry: ")
		if profName == 'cancel':
			self._GOTOMENU(menu=self._prevMenu)
		# perform drawing here!!!
		self._GOTOMENU(menu=self._prevMenu)

	def listAll(self):
		ind = 1
		for key,value in self._datasets.items():
			print("Name: {}\n - Classifier profiles:".format(key))
			for key in value.graphProfiles:
				print("   {}. {} - '{}'".format(ind,value.graphProfileNameTypes[key],key))
				ind = ind + 1

	def invalidSelect(self):
		print ("ERR: invalid menu choice.")
		self._GOTOMENU(menu=self._currentMenu)

	def invalidMenu(self):
		print ("ERR: invalid menu choice.")
		self._GOTOMENU(menu=self._currentMenu)

	def prevMenu(self):
		self._GOTOMENU(menu=self._prevMenu)

	def exitMenu(self):
		confirm = input("Are you sure you want to exit? (yes/no): ")
		if confirm.lower() == "yes":
			print ("Bye!")
			sys.exit()
		self._GOTOMENU(menu=self._currentMenu)

if __name__ == "__main__":
	iris=datasets.load_iris()
	bc=datasets.load_breast_cancer()
	d1="iris"
	d2="breast_cancer"
	interface = Interface(data1=iris,data2=bc,name1=d1,name2=d2)
	print("Initiated program")
	interface.startUp()