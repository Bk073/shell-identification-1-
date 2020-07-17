from models import base_model
from data.data_generator import train_data_generator, valid_data_generator, create_dataframe
import pandas as pd
import glob as glob
import  os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def train():
    batch_size = 128
    epochs = 5
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    img_dir = '/home/atlas/Atlas/Bishwa/data/image'
    input_shape = (150, 150, 3)
 
    df = create_dataframe(img_dir)
    train_data, valid_data = train_test_split(df, test_size=0.2,stratify=df['label'])
    train_data_gen = train_data_generator(train_data, batch_size = 128, IMG_HEIGHT = 150, IMG_WIDTH = 150, img_dir=img_dir)
    model = base_model.BaseModel(input_shape, train_data_gen)
    model.model()
    model.compile_model()
    model.fit(epochs)
    model.save()
    model.training_log()

#  '/home/atlas/Atlas/Bishwa/data/image'
# /home/bishwa/G/shell-identification/data/new_shell_images_2nd