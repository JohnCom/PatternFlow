

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical
import project as p

def load_data() :
    
    return [([0], [1]), ([2], [3])] #until load

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    num_classes = 4
    
    #creating a validation split
    num_images = len(train_images)
    num_labels = len(train_labels)
    
    val_split = 20 #20% for a multiple of 8
    num_val_images = int(num_images * (val_split/100))
    val_images = train_images[ : num_val_images] 
    val_labels = train_labels[ : num_val_images]
    
    train_images = train_images[num_val_images :]
    train_labels = train_labels[num_val_images :]
    
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    
    model = p.UNET()
    
    model.evaluate(test_images)
    
    model.fit(train_images, train_labels,
              validation_data=(val_images, val_labels),
              epochs=epochs, verbose=1, workers=4)