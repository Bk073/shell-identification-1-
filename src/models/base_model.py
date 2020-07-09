from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from data.data_generator import train_data_generator, valid_data_generator
from matplotlib import pyplot as plt

class BaseModel(object):
        def __init__(self, train_data_gen=None, valid_data_gen=None, test_data_gen=None):
            # self.input_shape = input_shape
            self.train_data_gen = train_data_gen
            self.valid_data_gen = valid_data_gen
            self.test_data_gen = test_data_gen
    
        def model(self):
            self.base_model = Sequential()
            self.base_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',))
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
        
        def fit(self, epochs):
            self.history = self.base_model.fit(self.train_data_gen,steps_per_epoch=self.train_data_gen.n//self.train_data_gen.batch_size,epochs=epochs)
        
        def save(self):
            self.base_model.save('../../models/base_model.h5')

        def eval(self):
            loss, accuracy = self.base_model.evaluate(self.test_data_gen)
            return loss, accuracy

        def training_log(self):
            acc = history.history['accuracy']
            loss=history.history['loss']
            epochs_range = range(len(acc))

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Accuracy')
            # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.legend(loc='lower right')
            plt.title('Accuracy and loss')
        
        def load_model(self):
            self.base_model = load_model('../../models/base_model.h5')