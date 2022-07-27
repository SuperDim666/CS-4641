from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(object):
    def __init__(self):
        # change these to appropriate values

        self.batch_size = 64
        self.epochs = 20
        self.init_lr= 1e-3 #learning rate
        
        #Higher batch size, slower learning but converge to more stable
        #Large epoch could overfit, Small epoch could underfit

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        #TODO: implement this
        #Input shape: 32, 32, 3
        self.model = Sequential()
        #Conv2D 1, shape: 32, 32, 8
        self.model.add(Conv2D(
            8, 
            kernel_size=(3, 3),
            input_shape=(32, 32, 3),
            padding="same"
        ))
        #Activation
        self.model.add(LeakyReLU(alpha=0.1))
        
        #Conv2D 2, shape: 32, 32, 32
        self.model.add(Conv2D(
            32, 
            kernel_size=(3, 3),
            padding="same"
        ))
        #Activation
        self.model.add(LeakyReLU(alpha=0.1))
        
        #Maxpool 1, shape: 16, 16, 32
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #Droput 1, rate = 0.30
        self.model.add(Dropout(0.30))
        
        #Conv2D 3, shape: 16, 16, 32
        self.model.add(Conv2D(
            32, 
            kernel_size=(3, 3),
            padding="same"
        ))
        #Activation
        self.model.add(LeakyReLU(alpha=0.1))
        
        #Conv2D 4, shape: 16, 16, 64
        self.model.add(Conv2D(
            64, 
            kernel_size=(3, 3),
            padding="same"
        ))
        #Activation
        self.model.add(LeakyReLU(alpha=0.1))
        
        #Maxpool 2
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #Dropout 2, rate = 0.30
        self.model.add(Dropout(0.30))
        #Flatten
        self.model.add(Flatten())
        
        #FC1, Dense 256
        self.model.add(Dense(256))
        #Activation
        self.model.add(LeakyReLU(alpha=0.1))
        
        #Droput 3, rate = 0.5
        self.model.add(Dropout(0.5))
        
        #FC2, Dense 128
        self.model.add(Dense(128))
        #Activation
        self.model.add(LeakyReLU(alpha=0.1))
        
        #Droput 4, rate = 0.5
        self.model.add(Dropout(0.5))
        #FC3, Dense 10, softmax
        self.model.add(Dense(10))
        #Activation
        self.model.add(Activation(tf.keras.activations.softmax))
        
        return self.model

    
    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model

        #TODO: implement this
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        return self.model
