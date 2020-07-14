from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

class Mobile_net(Model):
    def __init__(self):
        super(Mobile_net, self).__init__()
        self.base_model = MobileNet(weights='imagenet',include_top=False, pooling="avg", input_shape=(224,224,3))
        # self.global_pool = GlobalAveragePooling2D()
        self.dense_1 = Dense(1024, activation='relu')
        self.dense_2 = Dense(512, activation='relu')
        self.dense_3 =  Dense(7894, activation='softmax')

    
    def call(self, x):
        x = self.base_model(x)
        # x = self.global_pool(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x