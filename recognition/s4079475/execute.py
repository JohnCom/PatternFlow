

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical, Sequence
import project as p
import model as m
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import random

def encode_y(y):
    
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    return y

# A class that can be parsed in to the fit model parameter
# to have its functions called natively by fit model. 
#code ref
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

class ISICs2018Sequence(Sequence):
    
    def __init__(self, x, y, batchsize):
        
        self.x = x
        self.y = y
        self.batchsize = batchsize
        
    def __len__(self):
        
        return math.ceil(len(self.x) / self.batchsize)
    
    def __getitem__(self, id):
        
        x_names = self.x[id * self.batchsize:(id + 1) * self.batchsize]
        y_names = self.y[id * self.batchsize:(id + 1) * self.batchsize]
        
        x_batch = list()
        y_batch = list()
        
        for name in x_names :
            
            file_name = name[:len(name) - 4]
            
            train_image = np.asarray(Image.open("ISIC2018_Task1-2_Training_Input_x2/" + file_name + ".jpg").resize((256, 192))) / 255.0
        
            x_batch.append(train_image)
        
            ground_image = np.asarray(Image.open("ISIC2018_Task1_Training_GroundTruth_x2/" + file_name + "_segmentation" + ".png").resize((256, 192))) / 255.0 
           
            y_batch.append(ground_image)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        
        y_batch = encode_y(y_batch)
        
        return x_batch, y_batch

def load_data(directory, seed) :
    
    loaded = list()
    
    for file in os.listdir(directory) :
    
        if file != "ATTRIBUTION.txt" and file != "LICENSE.txt" :
            
            loaded.append(file)

    random.seed(seed)
    random.shuffle(loaded)

    return loaded

def show_images(test, model) :
    
    testlength = len(test)
    rand = random.randint(0, testlength)
    x, y = test.__getitem__(rand)
    prediction = model.predict(x)
    plt.figure(figsize=(10, 10))
    
    for i in range(8):
        plt.subplot(10, 10, i*10+2)
        plt.imshow(x[i])
        plt.title("Original Image", size=4)
        plt.subplot(10, 10, i*10+4)
        plt.imshow(tf.argmax(prediction[i], axis=2))
        plt.title("Model Output", size=4)
        plt.subplot(10, 10, i*10+6)
        plt.imshow(tf.argmax(y[i], axis=2))
        plt.title("What it should be", size=4)
    plt.show()

if __name__ == "__main__":

    seed = random.random()
    train = load_data("D:/3710sets/2018/ISIC2018_Task1-2_Training_Input_x2", seed)
    #test = load_data("D:/3710sets/ISIC_2019_Test_Input/ISIC_2019_Test_Input", seed)
    ground = load_data("D:/3710sets/2018/ISIC2018_Task1_Training_GroundTruth_x2", seed)
    
    #creating a validation split
    #num_images = len(train)
    #num_labels = len(test_names)
    
    #val_split = 20 #20% for a multiple of 8
    #num_val_images = int(num_images * (val_split/100))
    #val_images = train[ : num_val_images] 
    #val_seg = ground[ : num_val_images]
    
    #train_images = train[num_val_images :]
    #train_seg = ground[num_val_images :]
     
    #train_set = tf.data.Dataset.from_tensor_slices((train_images, train_seg))
    #test_set = tf.data.Dataset.from_tensor_slices((val_images, val_seg))    
    
    #train_set = train_set.map(map_images)
    #test_set = test_set.map(map_images)

    
    #suffle datasets?

    #train_labels = to_categorical(train_labels, num_classes)
    #test_labels = to_categorical(test_labels, num_classes)
    #val_labels = to_categorical(val_labels, num_classes)
    
    #print(len(train_images), "train shape")
    #print(len(train_seg), "train shape")    
    
    #print(train_images[:3], "three train images")
    #print(train_seg[:3], "three seg images")
       
    #view_images(train_images, 3)
    #plt.show()

    train_images, test_images, ground_images, ground_test = train_test_split(train, ground, test_size=0.20, random_state=50)

    train_images, val_images, ground_images, ground_val = train_test_split(train_images, ground_images, test_size=0.20, random_state=50)

    train = ISICs2018Sequence(train_images, ground_images, 8)
    val = ISICs2018Sequence(val_images, ground_val, 8)
    test = ISICs2018Sequence(test_images, ground_test, 8)
    
    model = p.UNET()
    
    print(len(train),  "length train")
    
    model.fit(train,
          validation_data=val,
          epochs=5, verbose=1, workers=4)
    
    model.evaluate(test)
    
    show_images(test, model)