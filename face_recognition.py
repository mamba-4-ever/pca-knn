import cv2
import numpy as np
import os
from sklearn import neighbors
import tkinter
from tkinter import filedialog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt

# load face database
#load data

def loadimages(data):
    
    '''
    data:train content
    images:[m,height,width]
    m sample number
    height
    width
    name
    label
    '''
    
    images = []
    labels = []
    names =[]
    label=1
    # Read all folders where photos are located
    for subdirname in os.listdir(data):
        subjectpath = os.path.join(data,subdirname)
        if os.path.isdir(subjectpath):
            
            #photos of each person in a folder
            names.append(subdirname)
            for filename in os.listdir(subjectpath):
                imgpath = os.path.join(subjectpath,filename)
                img = cv2.imread(imgpath,0)  #OR cv2.IMREAD_GRAYSCALE
                images.append(img)
                labels.append(label)
            label = label+1
    images = np.asarray(images) #Turn the list into an array, one picture corresponds to one row of the array
    labels =np.asarray(labels)
    return images,labels,names

class face(object):
    def __init__(self,dsize=(46,56)):
        '''
        dsize：Image preprocessing size
        '''
        self._dsize = dsize

    def _prepare(self,images):
        '''
        Image preprocessing, histogram equalization
        images：Training set data, grayscale pictures
        [m,height,width] m sample num, height width
        return processed data[m,n]
        feature num=dsize[0]*dsize[1]
        '''
        new_images = []
        for image in images:
            re_img = cv2.resize(image,self._dsize)
        #Histogram equalization
            hist_img = cv2.equalizeHist(re_img)
        #convert the data to one row
            hist_img = np.reshape(hist_img,(1,-1))
            new_images.append(hist_img)
        new_images = np.asarray(new_images)#list to array
        return np.squeeze(new_images)

if __name__=='__main__':
    Face = face()
    #import the training set
    data_train = "C:\\Users\\guoming5\\Desktop\\p_file\\PCA-KNN\\image datebase_new\\train set"
    x_train,y_train,names = loadimages(data_train)
    x_trainNEW = np.squeeze(Face._prepare(x_train))

    pca = PCA(n_components=200).fit(x_trainNEW)
    x_trainreduced = pca.transform(x_trainNEW)
    clf = knn(n_jobs=-1,n_neighbors=1,weights='uniform')
    clf.fit(x_trainreduced, y_train)

    #Import testing set to verify classification accuracy
    data_test = "C:\\Users\\guoming5\\Desktop\\p_file\\PCA-KNN\\image datebase_new\\test set"
    x_test,y_test,names = loadimages(data_test)
    x_testNEW = np.squeeze(Face._prepare(x_test))
    x_testreduced = pca.transform(x_testNEW)

    #print(clf.score(x_trainreduced, y_train))
    print("PCA+KNN................")
    print(clf.score(x_testreduced, y_test))

    #compute and show average face
    train = np.genfromtxt('train.csv', delimiter=',').astype(int)
    x_train=train[:,:19200]
    x_trainAverage = np.mean(x_train, axis=0)
    x_trainAverage1 = x_trainAverage
    x_trainAverage = x_trainAverage.reshape(120,160)
    plt.imshow(x_trainAverage, cmap='gray', vmin=0, vmax=255)
    plt.show()

    #reconstruct face
    pca = PCA(n_components=200).fit(x_train)
    x_trainreduced = pca.transform(x_train)
    eigen_faces = pca.components_
    total = sum(((eigen_faces.T)*x_trainreduced[258]).T)
    total = total+x_trainAverage1
    total = total.reshape((120,160))
    plt.imshow(total, cmap='gray', vmin=0, vmax=255)
    plt.show()