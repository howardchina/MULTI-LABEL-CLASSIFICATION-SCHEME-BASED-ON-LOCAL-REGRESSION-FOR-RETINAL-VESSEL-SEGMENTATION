###################################################
#
#   Script to execute the prediction
#
##################################################
import os, sys
import configparser as ConfigParser

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################
#Python
import numpy as np
import ConfigParser
from math import ceil
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import cv2
import sys

def my_PreProc(data):
    assert(len(data.shape)==3)
    assert (data.shape[2]==3)  #Use the original images
    #black-white conversion)
    img = rgb2gray(data)
    #my preprocessing:
    img = dataset_normalized(img)
    img = clahe_equalized(img)
    img = adjust_gamma(img, 1.2)
    img = img/255.  #reduce to 0-1 range
    return img

def rgb2gray(rgb):
    assert (len(rgb.shape)==3)  #4D arrays
    assert (rgb.shape[2]==3)
    bn_img = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    return bn_img

def dataset_normalized(img):
    assert (len(img.shape)==2)  #2D arrays
    img_normalized = np.empty(img.shape)
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img-img_mean)/img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255
    return img_normalized

def clahe_equalized(img):
    assert (len(img.shape)==2)  #2D arrays
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalized = clahe.apply(np.array(img, dtype = np.uint8))
    return img_equalized

def adjust_gamma(img, gamma=1.0):
    assert (len(img.shape)==2)  #2D arrays
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    
    new_img = cv2.LUT(np.array(img, dtype = np.uint8), table)
    return new_img

def matchsize(img,strip=5,size=48):
    h,w=img.shape
    new_h=h
    if (h-size)%strip != 0:
        new_h+=strip-(h-size)%strip
    new_w=w
    if (w-size)%strip != 0:
        new_w+=strip-(w-size)%strip
    img_new=np.zeros((new_h,new_w))
    img_new[0:h,0:w]=img
    print 'new h=%d w=%d' % (new_h,new_w)
    return img_new
    
def crop(img, strip=5, size=48):
    h,w=img.shape
    assert((h-size)%strip==0 and (w-size)%strip==0)
    imgs=np.zeros((((h-size+strip)/strip)*((w-size+strip)/strip),1,size,size))
    cnt=0
    for y in range(0,h-size+strip,strip):
        for x in range(0,w-size+strip,strip):
            imgs[cnt,0]=img[y:y+size,x:x+size]
            cnt+=1
    return imgs



#Load the saved model
model = model_from_json(open('architecture.json').read())
#model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
model.load_weights('finetune-weights-01-0.04.hdf5')
#Calculate the predictions

img=cv2.imread('01_test.tif')
img=my_PreProc(img)
strip=24
size=48
new_img=matchsize(img,strip=strip,size=size)
imgs=crop(new_img,strip=strip,size=size)
print imgs.shape

predictions = model.predict(imgs, batch_size=128, verbose=1)
print "predicted images size :"
print predictions[0].shape
print predictions[1].shape

def overlap(patches,h,w,strip=5,size=48):
    img=np.zeros((h,w))
    cnt_img=np.zeros((h,w))
    cnt=0
    for y in range(0,h-size+strip,strip):
        for x in range(0,w-size+strip,strip):
            img[y:y+size,x:x+size]+=patches[cnt]
            cnt_img[y:y+size,x:x+size]+=1
            cnt+=1
    #print h,w
    #print cnt
    #print patches.shape
    print np.min(cnt_img)
    assert(cnt==patches.shape[0])
    return img/cnt_img


cls=2
prediction_patches=np.reshape(predictions[0][:,:,cls],(predictions[0].shape[0],48,48))
bigVesselCenter=overlap(prediction_patches,new_img.shape[0],new_img.shape[1],strip=strip,size=size)
cls=3
prediction_patches=np.reshape(predictions[0][:,:,cls],(predictions[0].shape[0],48,48))
bigVesselEdge=overlap(prediction_patches,new_img.shape[0],new_img.shape[1],strip=strip,size=size)
cls=4
prediction_patches=np.reshape(predictions[0][:,:,cls],(predictions[0].shape[0],48,48))
smallVessel=overlap(prediction_patches,new_img.shape[0],new_img.shape[1],strip=strip,size=size)

#print np.min(prediction_patches)

#cv2.imwrite('./BC_prediction.png',bigVesselCenter*255)
#cv2.imwrite('./BE_prediction.png',bigVesselEdge*255)
#cv2.imwrite('./S_prediction.png',smallVessel*255)

import cv2
from PIL import Image
import numpy as np

def VThin(image,array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                if image[i,j] == 0  and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    
def HThin(image,array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1   
                if image[i,j] == 0 and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    
def Xihua(image,array,num=10):
    iXihua = np.zeros(image.shape)
    iXihua = image
    for i in range(num):
        VThin(iXihua,array)
        HThin(iXihua,array)
    return iXihua

def Two(image):
    w = image.shape[1]
    h = image.shape[0]
    size = (w,h)
    iTwo = np.zeros(image.shape)
    
    for i in range(h):
        for j in range(w):
            iTwo[i,j] = 0 if image[i,j] < 200 else 255
    return iTwo


array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]
def thinner(image):
    image = (image==0)*255
    iTwo = Two(image)
    iThin = Xihua(iTwo,array)
    newImage = (iThin==0)
    return newImage

EH=bigVesselCenter+bigVesselEdge+np.sqrt(smallVessel)
sk = thinner(EH>=0.5)
final=(sk+(bigVesselCenter+bigVesselEdge+smallVessel)>=0.5)>0
#cv2.imwrite('./enhancement.png',EH*255)
#cv2.imwrite('./sk.png',sk*255)
cv2.imwrite('./final.png',final[0:img.shape[0],0:img.shape[1]]*255)
print 'Done'