#!/usr/bin/env python
"""
    File name: softmax_regression.py
    Author: skconan
    Date created: 2019/02/07
    Python Version: 3.5
"""
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
import pandas as pd
import random
import csv
import keras
import random
from keras.models import Model
from keras.layers import *
from keras import optimizers
import time

def get_dataset():
    print("Get Data")
    # lebel, images
    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []
    
    dataset = pd.read_csv(r"C:\Users\skconan\Desktop\underwater_object_detection\dataset.csv")
    dataset = np.array(dataset)
    tmp = pd.read_csv(r"C:\Users\skconan\Desktop\underwater_object_detection\dataset_noise.csv")
    dataset2 = np.array(tmp)
    
    for d in dataset:
        # print(d)
        img = d[1:]
        label = [0,0,0]
        index = 0
        for i in str(int(d[0])):
            if i == '0':
                index = 0
            else:
                index += 2**(int(i)-1)
        label[int(index)] = 1
        
        if random.uniform(0, 1) <= .8: 
            train_data_x.append(img)
            train_data_y.append(label)
        else:
            test_data_x.append(img)
            test_data_y.append(label)

    for d in dataset2:
        # print(d)
        img = d[1:]
        label = [0,0,0]
        index = 0
        for i in str(int(d[0])):
            if i == '0':
                index = 0
            else:
                index += 2**(int(i)-1)
        label[int(index)] = 1
        
        if random.uniform(0, 1) <= .8: 
            train_data_x.append(img)
            train_data_y.append(label)
        else:
            test_data_x.append(img)
            test_data_y.append(label)
    print("end")
    return  np.array(train_data_x), \
            np.array(train_data_y), \
            np.array(test_data_x), \
            np.array(test_data_y)

def preprocessing_data():
    """
        onehot
    """
    print("preprocessing data")
    directory = os.listdir(CONST.PATH_DOT_IMAGE)
    f = open(CONST.PATH_DATASET+"\dot_image_train.csv",'w+')
    f1 = open(CONST.PATH_DATASET+"\dot_image_test.csv",'w+')
    for d in directory:
        path = CONST.PATH_DOT_IMAGE + "\\" + d
        images_name = os.listdir(path)
        for name in images_name:
            rand = random.random()

            
            img = cv.imread(path+"\\"+name,0)
            if img is None:
                continue
            if rand <= 0.7:
                for v in img.ravel():
                    f.write(str(v/255.)+", ")
                f.write(d)
                f.write("\n")
            else:
                for v in img.ravel():
                    f1.write(str(v/255.)+", ")
                f1.write(d)
                f1.write("\n")
            # exit(0)
    print("end")

def softmax_regression():
    X_train, y_train, X_cv, y_cv = get_dataset()
    print(len(X_train))
    print(len(y_train))

    # Input Parameters
    n_input = 36784 # number of features
    n_hidden_1 = 800
    n_hidden_2 = 400
    n_hidden_3 = 200
    # n_hidden_4 = 200
    num_digits = 3

    Inp = Input(shape=(n_input,))
    x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
    x = Dropout(2)(x)
    x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
    x = Dropout(2)(x)
    x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
    x = Dropout(2)(x)
    # x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
    # x = Dropout(0.3)(x)
    # x = Dense(n_hidden_5, activation='relu', name = "Hidden_Layer_5")(x)
    # x = Dropout(0.2)(x)
    # x = Dense(n_hidden_6, activation='relu', name = "Hidden_Layer_6")(x)
    # x = Dropout(0.2)(x)
    output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

    model = Model(Inp, output)
    model.summary()

    # Insert Hyperparameters
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 20
    adam = keras.optimizers.Adam(lr=learning_rate)

    # We rely on the plain vanilla Stochastic Gradient Descent as our optimizing methodology
    with tf.device('/device:GPU:0'):   
        model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])
        
        history1 = model.fit(X_train, y_train,
                        batch_size = batch_size,
                        epochs = training_epochs,
                        verbose = 2,
                        validation_data=(X_cv, y_cv))
        
        model.save(r'.\ep'+ str(training_epochs) +'_bch'+ str(batch_size) +'_r'+ str(learning_rate)+'.h5') 
    # 00 set 05 input 00
    # 00 set 05 input 01
if __name__ == "__main__":
    # preprocessing_data()
    softmax_regression()
