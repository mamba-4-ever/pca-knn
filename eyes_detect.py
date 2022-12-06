# coding: utf-8
import cv2
import dlib
import sys
import numpy as np
import os
import matplotlib.pyplot as plt


people=os.listdir("C:\\Users\\guoming5\\Desktop\\p_file\\PCA-KNN\\image datebase_new\\train set")
dir="C:\\Users\\guoming5\\Desktop\\p_file\\PCA-KNN\\image datebase_new\\train set\\"+people[1]
fname=os.listdir(dir)
basedir=dir+"\\"+fname[0]



SCALE_FACTOR = 1 
PREDICTOR_PATH = "C:\\Users\\guoming5\\Desktop\\p_file\\PCA-KNN\\tools\\shape_predictor_5_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
	rects = detector(im, 1)
	'''
	if len(rects) > 1:
		raise TooManyFaces
'''
	if len(rects) == 0:
		print("0rect")
		return []

	return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def read_im_and_landmarks(fname):
	temim = cv2.imread(fname, cv2.IMREAD_COLOR)
	im = cv2.cvtColor(temim, cv2.COLOR_BGR2GRAY)
	#im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
	#					 im.shape[0] * SCALE_FACTOR))
	im = cv2.resize(im,(120,160))
	s = get_landmarks(im)

	return im, s
	
def warp_im(im, M, dshape):
	output_im = np.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im,
				   M[:2],
				   (dshape[1], dshape[0]),
				   dst=output_im,
				   borderMode=cv2.BORDER_TRANSPARENT,
				   flags=cv2.WARP_INVERSE_MAP)
	return output_im

def annotate_landmarks(im, landmarks):
   
	im = im.copy()
	for idx, point in enumerate(landmarks):
		pos = (point[0, 0], point[0, 1])
		cv2.putText(im, str(idx), pos,
					fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
					fontScale=0.4,
					color=(0, 0, 255))
		cv2.circle(im, pos, 3, color=(0, 255, 255))
	return im
	
def transformation_from_points(points1, points2):
	points1 = points1.astype(np.float64)
	points2 = points2.astype(np.float64)

	c1 = np.mean(points1, axis=0)
	c2 = np.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2

	s1 = np.std(points1)
	s2 = np.std(points2)
	points1 /= s1
	points2 /= s2

	U, S, Vt = np.linalg.svd(points1.T * points2)
	R = (U * Vt).T

	return np.vstack([np.hstack(((s2 / s1) * R,
									   c2.T - (s2 / s1) * R * c1.T)),
						 np.matrix([0., 0., 1.])])
						 
def face_Align(Base_path,cover_path):
	im1, landmarks1 = read_im_and_landmarks(Base_path)  
	im2, landmarks2 = read_im_and_landmarks(cover_path)  
	
	if len(landmarks1) == 0 or len(landmarks2) == 0:
		#raise ImproperNumber("Faces detected is no face!")
		print("No face detected")
		return im2
		
	elif len(landmarks1) > 1 & len(landmarks2) > 1:
		#raise ImproperNumber("Faces detected is more than 1!")
		print("No face detected11")
		return im2
	
	else:
		M = transformation_from_points(landmarks1, landmarks2)
		warped_im2 = warp_im(im2, M, (120,160))
		return warped_im2


train=np.empty((0,19201), int)
test=np.empty((0,19201), int)

'''
train=np.empty((0,19201), int)
test=np.empty((0,19201), int)
'''
for i in range(len(people)):
	dir="C:\\Users\\guoming5\\Desktop\\p_file\\PCA-KNN\\image datebase_new\\train set\\"+people[i]
	fname=os.listdir(dir)
	for j in range(12):
		print("processing people"+people[i]+"---photo"+fname[j])
		fdir=dir+"\\"+fname[j]
		img=face_Align(basedir,fdir)
		#warped_mask=face_Align(basedir,fdir)
		#size = (120, 160)
		#img=cv2.resize(warped_mask,size)
		#img=autocontrast(img)
		
		
		a=np.asarray(img).reshape(1,-1)	 
		y=np.matrix([[i]])
		a=np.append(a,y,axis=1)
		if j>11:
			test=np.append(test,a,axis=0)
		else:
			train=np.append(train,a,axis=0)
			
print(train.shape,test.shape)
np.savetxt('train.csv', train, delimiter=',')
np.savetxt('test.csv', test, delimiter=',')