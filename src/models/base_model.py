from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from data.data_generator import train_data_generator, valid_data_generator

class BaseModel(object):
        def __init__(self, input_shape, train_data_gen, valid_data_gen=None):
            self.input_shape = input_shape
            self.train_data_gen = train_data_gen
            self.valid_data_gen = valid_data_gen
    
        def model(self):
            self.base_model = Sequential()
            self.base_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.input_shape))
            self.base_model.add(Conv2D(64, (3, 3), activation='relu'))
            self.base_model.add(MaxPooling2D(pool_size=(2, 2)))
            self.base_model.add(Dropout(0.25))
            self.base_model.add(Conv2D(128, (3, 3), activation='relu'))
            self.base_model.add(MaxPooling2D(pool_size=(2, 2)))
            self.base_model.add(MaxPooling2D(pool_size=(2, 2)))
            self.base_model.add(Conv2D(128, (3, 3), activation='relu'))
            self.base_model.add(MaxPooling2D(pool_size=(2, 2)))
            self.base_model.add(MaxPooling2D(pool_size=(2, 2)))
            self.base_model.add(Flatten())
            self.base_model.add(Dropout(.6))
            # self.base_model.add(Dense(128, activation='relu'))
 
            self.base_model.add(Dense(7894, activation='softmax'))
        
            # return self.base_model
        
        def compile_model(self):
            self.base_model.compile(loss="categorical_crossentropy",
                    optimizer='adam',
                    metrics=['accuracy'])
        
        def fit(self):
            self.history = self.base_model.fit(self.train_data_gen,steps_per_epoch=self.train_data_gen.n//self.train_data_gen.batch_size,epochs=2)
        
