import pandas as pd
import glob as glob
import  os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
 
def create_dataframe(path):
   image_dict = {}
   image_dict['image'] = []
   image_dict['label'] = []
  # for img in tqdm(os.listdir(img_dir)):
 #   label = label_img(img)
 #   image_dict['label'].append(label)
 
   for imagepath in glob.glob(os.path.join(path, '*.jpg')):
       image = imagepath.split("/")[7]
       image_dict['image'].append(image)
       split = image.split('_')
       label = split[0] + ' ' + split[1]
       image_dict['label'].append(label)
   df = pd.DataFrame.from_dict(image_dict)
   return df
 
def train_data_generator(train_data, batch_size, img_dir, IMG_HEIGHT, IMG_WIDTH):
   train_image_generator = ImageDataGenerator(rescale=1./255)
   train_data_gen = train_image_generator.flow_from_dataframe(
                                                         dataframe = train_data,
                                                         x_col = "image",
                                                         y_col = "label",
                                                         batch_size=batch_size,
                                                          directory=img_dir,
                                                          shuffle=True,
                                                          target_size=(IMG_HEIGHT,
                                                          IMG_WIDTH),
                                                          class_mode='categorical')
   return train_data_gen
 
def valid_data_generator(valid_data, img_dir, IMG_HEIGHT, IMG_WIDTH):
   validation_image_generator = ImageDataGenerator(rescale=1./255)
   valid_data_gen = validation_image_generator.flow_from_dataframe(
                                                         dataframe = valid_data,
                                                         x_col = "image",
                                                         y_col = "label",
                                                          directory=img_dir,
                                                          shuffle=False,
                                                          target_size=(IMG_HEIGHT,
                                                          IMG_WIDTH),
                                                          class_mode='categorical')
   return valid_data_gen
 
def test_data_generator(test_data, img_dir, IMG_HEIGHT, IMG_WIDTH):
   test_image_generator = ImageDataGenerator(rescale=1./255)
   test_data_gen = test_image_generator.flow_from_dataframe(
                                                         dataframe = test_data,
                                                         x_col = "image",
                                                         y_col = "label",
                                                         directory=img_dir,
                                                         shuffle=False,
                                                         target_size=(IMG_HEIGHT,
                                                         IMG_WIDTH),
                                                         class_mode='categorical')
   return test_data_gen
   
if '__name__' == '__main__':
   batch_size = 128
   epochs = 5
   IMG_HEIGHT = 150
   IMG_WIDTH = 150
   img_dir = '../../data/new_shell_images_2nd'
 
   df = create_dataframe()
   train_data, valid = train_test_split(df, test_size=0.2,stratify=df['label'])
   valid_data, test_data = train_test_split(valid, test_size=0.1, stratify=valid['label'])
   train_data_generator = train_data_generator(train_data, batch_size = 128, IMG_HEIGHT = 150, IMG_WIDTH = 150, img_dir=img_dir)
