
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision

def dice(true, predicted):
    """

    This implements the metric for dice similarity.
    
    @param - true -- the true test dataset result / image result
    @param - predicted -- the model predicted result
    
    """
    
    #true = true[:, :, 1]
    #predicted = predicted[:, :, 1]
    
    true = tf.keras.backend.batch_flatten(true)
    predicted = tf.keras.backend.batch_flatten(predicted)
    predicted = tf.keras.backend.round(predicted)
    
    e_positive = tf.keras.backend.sum(true, axis= -1)
    p_positive = tf.keras.backend.sum(predicted, axis= -1)
    
    sum = tf.keras.backend.sum(predicted * true, axis = -1)
    
    false_neg = e_positive  - sum
    false_pos = p_positive - sum
    
    return (2.0 * sum)/(2 * sum + false_neg + false_pos)

def dice_foreground(true, predicted):

    """
    Only calculates dice similarity for foreground results (the parts of the image that are deemed problematic).
    
    @param - true -- the true test dataset result / image result
    @param - predicted -- the model predicted result
    
    """
    
    return dice(true[:,:,1], predicted[:,:,1])

def dice_background(true, predicted):
    
    """
    Only calculates dice similarity for background results (the parts of the image surrounding the problematic area).
    
    @param - true -- the true test dataset result / image result
    @param - predicted -- the model predicted result
    
    """
    
    return dice(true[:,:,0], predicted[:,:,0])

def feature_conv_layer(input, filters, stride=1, size=(3, 3)) :
    
    """
    This implements a standard convolution layer followed by ReLU layer 
    
    @param - input -- the input tensor
    @param - filters -- the number of convolution filters to apply.
    @param - stride -- the number of strides to use in the convolution. 
    @param - size -- the size of the kernel in the convolution
    
    """
    
    result = tf.keras.layers.Conv2D(filters, size, stride, padding='same')(input)
    result = tf.keras.layers.ReLU()(result)
    return result

def extraction_layer(input, filters) :

    """
    This implements the context module defined in Isensee et al. 
    With a slightly modified dropout hyperparameter.
    
    @param - input -- the input tensor
    @param - filters -- the number of convolution filters to apply.
    
    """
    
    result = feature_conv_layer(input, filters)
    result = feature_conv_layer(result, filters)
    result = tf.keras.layers.Dropout(0.2)(result)
    return tf.keras.layers.add([result, input])

def upsample_layer(input, filters, inputTwo) :
    
    """
    This implements the upsampling module defined in Isensee et al. 
    
    @param - input -- the input tensor
    @param - filters -- the number of convolution filters to apply.
    @param - inputTwo -- The previous input tensor to concatenate to the result.
    
    """
    
    result = tf.keras.layers.UpSampling2D((2, 2))(input)
    result = feature_conv_layer(result, filters)
    return tf.keras.layers.concatenate([result, inputTwo])

def downsample_layer(input) :

    """
    This implements a downsampling layer as defined in Isensee et al using MaxPooling. 
    
    @param - input -- the input tensor
    
    """
    
    result = tf.keras.layers.MaxPooling((2, 2))(input)
    return result


def localisation(input, filters) :
    """
    This implements the localisation module defined in Isensee et al. 
    
    @param - input -- the input tensor
    @param - filters -- the number of convolution filters to apply.
    
    """
    
    result = feature_conv_layer(input, filters)
    result = feature_conv_layer(result, filters, size=(1,1))
    return result

#Implements the improved UNET structure as defined in the paper by F. Isensee et al. 
def UNET () :

    """
    Implementation of the improved UNET model as defined in the paper by 
    Isensee et al. https://arxiv.org/abs/1802.10508v1
    
    It makes a couple of small hyperparameter modifications that have been deemed to have a small increase in performance based on the metric calculations used for dice similarity. 
    
    As the model in the paper suggests this implementation uses multiple context modules (defined in extraction_layer) together with downsampling layers followed by localisation_modules (defined in refine_layer) togetehr with upsampling modules. Segmentation is then done towards the end of the model and concatenation of different final level segmented convolutions is done to avoid learning loss.
    
    @returns --- an improved UNET model defined in the paper 

    """
    
    input = tf.keras.layers.Input(shape=(192, 256, 3))

    filters = 8    
    
    #Initial convolution
    current = feature_conv_layer(input, filters)
    
    #context module
    extract1 = extraction_layer(current, filters)
    filters = filters * 2
    #downsampling
    current = feature_conv_layer(extract1, filters, stride=2)
    
    #context     
    extract2 = extraction_layer(current, filters)
    filters = filters * 2
    #downsampling
    current = feature_conv_layer(extract2, filters, stride=2)
    
    extract3 = extraction_layer(current, filters)
    filters = filters * 2
    current = feature_conv_layer(extract3, filters, stride=2)    

    extract4 = extraction_layer(current, filters)
    filters = filters * 2
    current = feature_conv_layer(extract4, filters, stride=2)
    
    #final context
    extract = extraction_layer(current, filters)
    filters = filters / 2
    #upsample
    current = upsample_layer(extract, filters, extract4)  
    
    #localisation module
    current = localisation(current, filters)
    filters = filters/2
    #upsample
    current = upsample_layer(current, filters, extract3)

    current = localisation(current, filters)
    filters = filters/2
    segment = feature_conv_layer(current, filters, size=(1, 1))
    segment = tf.keras.layers.UpSampling2D((2, 2))(segment)
    current = upsample_layer(current, filters, extract2)
    
    current = localisation(current, filters)
    filters = filters/2
    segment2 = feature_conv_layer(current, 16, size=(1, 1))
    
    #segment concatentation
    segment2 = tf.keras.layers.add([segment, segment2])
    segment2 = tf.keras.layers.UpSampling2D((2, 2))(segment2)
    current = upsample_layer(current, filters, extract1)    

    result = feature_conv_layer(current, 32)
    segment3 = feature_conv_layer(result, 16, size=(1, 1))
    
    #segment concatenation
    segment3 = tf.keras.layers.add([segment3, segment2])
    
    #Activation softmax layer
    result = tf.keras.layers.Conv2D(filters, (1, 1), padding = "same", activation="softmax")(segment3)   
    
    model = tf.keras.Model(inputs=input, outputs=result)
    
    model.summary()
    
    model.compile(loss= tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy', dice])
                  
    return model             