# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:51:49 2022

@author: raj.yadav
"""

# 1 step is data preprocessing , so we convert the data in mumpy array(batch processing) and normalize it.

import cv2
import glob
import ntpath 
import fnmatch
import numpy as np

train = []
train_labels = []
test=[]
test_labels=[]
count_train=0
count_test=0

#for train data
files=glob.glob('/train/*.jpg')    
for filename in files:
    img=cv2.imread(filename)
    head, tail = ntpath.split(filename)
    img=cv2.resize(img, (244,244))
    train.append(img)
    if fnmatch.fnmatch(tail, '*bad*'):
        train_labels.append([0])
    else:
        train_labels.append([1])
train=np.array(train,dtype='float32')
train_labels=np.array(train_labels,dtype='float64')

#for test data
files=glob.glob('/test/*.jpg')    
for filename in files:
    img=cv2.imread(filename)
    head,tail=ntpath.split(filename)
    img=cv2.resize(img, (244,244))
    test.append(img)
    if fnmatch.fnmatch(tail, '*bad*'):
        test_labels.append([0])
    else:
        test_labels.append([1])
test=np.array(test,dtype='float32')
test_labels=np.array(test_labels,dtype='float64')


# now creating our model neural network
# from keras.models import load_model
# model=load_model("FaceQnet.h5")
# model.summary()
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout

INPUT_SHAPE=(244,244,3)

model=Sequential()
#what is input and input layer
# 1st convlutional layer
model.add(Conv2D(32,(3,3),activation=('relu'),input_shape=(244,244,3)))
model.add(MaxPool2D(pool_size=(2,2)))
# 2nd convolutional layer
model.add(Conv2D(64,(3,3),activation=('relu')))
model.add(MaxPool2D(pool_size=(2,2)))
# 3rd convolutional layer
model.add(Conv2D(128,(3,3),activation=('relu')))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(512,activation=('relu')))
model.add(Dense(1,activation=('sigmoid')))

model.summary()
# Compiling the model
model.compile(loss=('binary_crossentropy'),optimizer='rmsprop',metrics=('accuracy'))

#adding a checkpoint to save best model
import os
from keras.callbacks import ModelCheckpoint

MODEL_PATH='C:/Users/raj.yadav/Projects/Self_try/Binary_classifier/Models'
filepath=os.path.join(MODEL_PATH,"FaceQnet_new.h5")
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max',period=1)
callback_list=[checkpoint]

#model fitting
model.fit(train,train_labels,batch_size=32,epochs=50,validation_data=(test,test_labels),callbacks=callback_list)