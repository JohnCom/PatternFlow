
#project model file

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import mixed_precision



def feature_conv_layer(input, filters, stride=1, size=(3, 3)) :
    
    result = tf.keras.layers.Conv2D(filters, size, stride, padding='same')(input)
    result = tf.keras.layers.LeakyReLU(alpha=0.01)(result)
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

def refine_layer(input, filters) :
    
    result = feature_conv_layer(input, filters)
    result = feature_conv_layer(result, filters, size=(1,1))
    return result

def UNET () :

    input = tf.keras.layers.Input(shape=(192, 256, 3))

    filters = 8    
    current = feature_conv_layer(input, filters)
    
    depth = 4
    
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
    segment = feature_conv_layer(current, 16, size=(1, 1))
    segment = tf.keras.layers.UpSampling2D((2, 2))(segment)
    filters = filters/2
    current = upsample_layer(current, filters, extract2)
    
    current = refine_layer(current, filters)
    segment2 = feature_conv_layer(current, 16, size=(1, 1))
    segment2 = tf.keras.layers.add([segment, segment2])
    segment2 = tf.keras.layers.UpSampling2D((2, 2))(segment2)
    filters = filters/2
    current = upsample_layer(current, filters, extract1)    

    result = feature_conv_layer(current, 32)
    segment3 = feature_conv_layer(result, 16, size=(1, 1))
    segment3 = tf.keras.layers.add([segment3, segment2])
    
    result = tf.keras.layers.Conv2D(2, (1, 1), padding = "same", activation="softmax")(segment3)
    
    model = tf.keras.Model(inputs=input, outputs=result)
    model.compile(loss= tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])
                  
    return model              

def improved_UNET() :

    batchSize = 32
    num_classes = 10
    epochs = 20
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    #creating a validation split
    num_images = len(train_images)
    num_labels = len(train_labels)
    
    val_split = 20 #20% for a multiple of 8
    num_val_images = int(num_images * (val_split/100))
    val_images = train_images[ : num_val_images] 
    val_labels = train_labels[ : num_val_images]
    
    train_images = train_images[num_val_images :]
    train_labels = train_labels[num_val_images :]
    
    shape = np.shape(train_images)   
    input_shape = shape[1:]
    
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    val_images = val_images.astype('float32') / 255
    
    #augmentation of data -- single horizontal flip
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        ])
    
    train_images = data_augmentation(train_images)
    
    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_images)
    
    train_labels = np_utils.to_categorical(train_labels, num_classes)
    test_labels = np_utils.to_categorical(test_labels, num_classes)
    val_labels = np_utils.to_categorical(val_labels, num_classes)

    #generate the model
    model = resNetLogic(input_shape, resnetDepth, num_classes);
    
    #compile the model
    model.compile(loss= tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    #learning rate scheule for model fitting 
    lr_scheduler = LearningRateScheduler(lr_schedule)

    #try just using one
    #define reduction of lr
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]

    #execute model
    model.fit(train_images, train_labels, batch_size=batchSize,
                  validation_data=(val_images, val_labels),
                  epochs=epochs, verbose=1, workers=4,
                  callbacks=callbacks)
    
    #evaluate    
    scores = model.evaluate(test_images, test_labels, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1]) 