

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical
import project as p
import os
import matplotlib.pyplot as plt
import random

def load_data(directory, seed) :
    
    loaded = [file for file in os.listdir(directory) 
            if file != "ATTRIBUTION.txt" and file != "LICENSE.txt"]


    random.seed(seed)
    random.shuffle(loaded)

    return loaded

def load_segment(image_shape, segment_file):
    
    segment = tf.io.read_file(segment_file)
    segment = tf.image.decode_jpeg(segment, channels=1)
    segment = tf.image.resize(segment, image_shape)
    segment = tf.cast(segment, tf.float32)
    segment =  segment / 255.0
    return segment

def load_image(image_shape, image_file):

    print(image_file, "image file")
    print(image_shape, "image shape")

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_shape)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

def map_images(image_file, seg_file, image_shape=(192, 256)):
    
    image = load_image(image_shape, image_file)
    seg = load_segment(image_shape, seg_file)
    
    return image, seg

def view_images(dataset, n):
    
    plt.figure(figsize=(8, n*4))
    i = 0

    for img, label in dataset.take(n):
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(img)
        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(label, cmap='gray')
        i = i + 1

if __name__ == "__main__":

    seed = random.random()
    train = load_data("D:/3710sets/2018/ISIC2018_Task1-2_Training_Input_x2", seed)
    #test = load_data("D:/3710sets/ISIC_2019_Test_Input/ISIC_2019_Test_Input", seed)
    ground = load_data("D:/3710sets/2018/ISIC2018_Task1_Training_GroundTruth_x2", seed)
    
    num_classes = 4 #find this out?
    
    #creating a validation split
    num_images = len(train)
    #num_labels = len(test_names)
    
    val_split = 20 #20% for a multiple of 8
    num_val_images = int(num_images * (val_split/100))
    val_images = train[ : num_val_images] 
    val_seg = ground[ : num_val_images]
    
    train_images = train[num_val_images :]
    train_seg = train[num_val_images :]
     
    train_set = tf.data.Dataset.from_tensor_slices((train_images, train_seg))
    test_set = tf.data.Dataset.from_tensor_slices((val_images, val_seg))
    
    train_set = train_set.map(map_images)
    test_set = test_set.map(map_images)
    
    #for element in train_set :
        #print(element)

    #suffle datasets?

    #train_labels = to_categorical(train_labels, num_classes)
    #test_labels = to_categorical(test_labels, num_classes)
    #val_labels = to_categorical(val_labels, num_classes)

    view_images(train_set, 3)
    plt.show()
    
    model = p.UNET()
    
    model.evaluate(train_set)
    
    model.fit(train_names,
              validation_data=(val_images),
              epochs=epochs, verbose=1, workers=4)