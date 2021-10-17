
#project model file

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision

def dice(true, predicted):

    true = tf.keras.backend.flatten(true)
    predicted = tf.keras.backend.flatten(predicted)
    sum = tf.keras.backend.sum(predicted * true)
    return (2.0 * sum)/tf.keras.backend.sum(predicted + true)

def dice_foreground(true, predicted):
    
    return dice(true[:,:,1], predicted[:,:,1])

def dice_background(true, predicted):
    
    return dice(true[:,:,0], predicted[:,:,0])

def feature_conv_layer(input, filters, stride=1, size=(3, 3)) :
    
    result = tf.keras.layers.Conv2D(filters, size, stride, padding='same')(input)
    result = tf.keras.layers.ReLU()(result)
    return result

def extraction_layer(input, filters) :

    result = feature_conv_layer(input, filters)
    result = feature_conv_layer(result, filters)
    result = tf.keras.layers.Dropout(0.2)(result)
    return tf.keras.layers.add([result, input])

def upsample_layer(input, filters, inputTwo) :
    
    result = tf.keras.layers.UpSampling2D((2, 2))(input)
    result = feature_conv_layer(result, filters)
    return tf.keras.layers.concatenate([result, inputTwo])

def downsample_layer(input) :
    
    result = tf.keras.layers.MaxPooling((2, 2))(input)
    return result

def refine_layer(input, filters) :
    
    result = feature_conv_layer(input, filters)
    result = feature_conv_layer(result, filters, size=(1,1))
    return result

def UNET () :

    input = tf.keras.layers.Input(shape=(192, 256, 3))

    #although paper says start with 16 seem to get better results starting with 8? 
    filters = 8    
    current = feature_conv_layer(input, filters)
    
    extract1 = extraction_layer(current, filters)
    filters = filters * 2
    current = feature_conv_layer(extract1, filters, stride=2)

    extract2 = extraction_layer(current, filters)
    filters = filters * 2
    current = feature_conv_layer(extract2, filters, stride=2)
    
    extract3 = extraction_layer(current, filters)
    filters = filters * 2
    current = feature_conv_layer(extract3, filters, stride=2)    

    extract4 = extraction_layer(current, filters)
    filters = filters * 2
    current = feature_conv_layer(extract4, filters, stride=2)
    
    extract = extraction_layer(current, filters)
    filters = filters / 2
    current = upsample_layer(extract, filters, extract4)  
    
    #local 
    current = refine_layer(current, filters)
    filters = filters/2
    current = upsample_layer(current, filters, extract3)

    current = refine_layer(current, filters)
    filters = filters/2
    segment = feature_conv_layer(current, filters, size=(1, 1))
    segment = tf.keras.layers.UpSampling2D((2, 2))(segment)
    current = upsample_layer(current, filters, extract2)
    
    current = refine_layer(current, filters)
    filters = filters/2
    segment2 = feature_conv_layer(current, 16, size=(1, 1))
    segment2 = tf.keras.layers.add([segment, segment2])
    segment2 = tf.keras.layers.UpSampling2D((2, 2))(segment2)
    current = upsample_layer(current, filters, extract1)    

    result = feature_conv_layer(current, 32)
    segment3 = feature_conv_layer(result, 16, size=(1, 1))
    segment3 = tf.keras.layers.add([segment3, segment2])
    
    result = tf.keras.layers.Conv2D(filters, (1, 1), padding = "same", activation="softmax")(segment3)   
    
    model = tf.keras.Model(inputs=input, outputs=result)
    
    model.summary()
    
    model.compile(loss= tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy', dice_foreground])
                  
    return model             