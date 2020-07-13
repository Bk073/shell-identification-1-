from models import mobile-net
from data.data_generator import train_data_generator, valid_data_generator, create_dataframe
import pandas as pd
import glob as glob
import  os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def train():
    batch_size = 128
    epochs = 10
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    img_dir = '/home/atlas/Atlas/Bishwa/data/image'
    input_shape = (224, 224, 3)
 
    df = create_dataframe(img_dir)
    train_data, valid_data = train_test_split(df, test_size=0.2,stratify=df['label'])
    train_data_gen = train_data_generator(train_data, batch_size = batch_size, IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH, img_dir=img_dir)
    model = mobile-net.Mobile_net()
    model.model()
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    history = model.fit(train_data_gen,steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,epochs=epochs)
    model.save_weights('v1.00.mobile-net.h5')

#  '/home/atlas/Atlas/Bishwa/data/image'
# /home/bishwa/G/shell-identification/data/new_shell_images_2nd