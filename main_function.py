from load_data import load_data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

def DataScatterPlot(X,target,ax,target_names,feature_names):
	''' Function to plot a scatter plot of data
		target: output labels
		ax: axis to plot to
		target_names: for generating the legend
		feature_names: for generating the axis labels
		'''
	classes = (0,1)
	colors = ('blue','red','green','yellow','black')
    
	for cl, color in zip(classes,colors):
		ax.scatter(X[target==cl, 0], X[target==cl, 1],s=10, c=color,label=target_names[cl])
        
    # ax.set_aspect('equal')
	plt.xlabel(feature_names[0])
	plt.ylabel(feature_names[1])
	plt.legend()
	plt.show()


if __name__ == "__main__":
	#load_data()
	
	train = np.genfromtxt('train.csv', delimiter=',')
	test = np.genfromtxt('test.csv', delimiter=',')
	
	random.Random(6324).shuffle(train)
	
	train_x=train[:,:4096]
	train_y=train[:,4096]
	test_x=test[:,:4096]
	test_y=test[:,4096]
	'''
	X_scaler =StandardScaler().fit(test_x)
	
	reduced_train=X_scaler.transform(train_x)
	reduced_test=X_scaler.transform(test_x)
	'''
	
	#print(train_y)
	
	print("PCA.............")
	
	pca = PCA(n_components=64).fit(train_x)
	reduced_train=pca.transform(train_x)
	reduced_test=pca.transform(test_x)
	'''
	target_names=['cf','r']
	feature_names=['cf','r']
	fig, ax = plt.subplots()
	DataScatterPlot(reduced_train,train_y,ax,target_names,feature_names)
	
	'''
	
	print("KNN................")
	clf = knn(n_jobs=-1,n_neighbors=6,weights='uniform')
	clf.fit(reduced_train, train_y)
	print(clf.score(reduced_test, test_y))
	
	'''
	
	clf = SVC()
	clf.fit(reduced_train, train_y)
	accuracy = clf.score(reduced_test, test_y)
	print("The accuracy of the classifier is: ", accuracy )
	'''
	'''
	print("Decision Tree..............................")
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(reduced_train, train_y)
	print(clf.score(reduced_test, test_y))
	'''
	'''
	svm = SVC(gamma='auto', kernel='linear', probability=True)
	svm.fit(reduced_train, train_y) 
	y_pred = svm.predict(reduced_test)'''

	#Evaluation
	'''
	precision = metrics.accuracy_score(y_pred, test_y) * 100
	print("Accuracy with SVM: {0:.2f}%".format(precision))
    '''
	'''
	print(clf.score(reduced_test, test_y))
	'''
	#print(reduced_train)