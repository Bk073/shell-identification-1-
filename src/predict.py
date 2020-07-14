from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from models.mobile_net import Mobile_net
import json

def load_label():
    label_path = '/home/atlas/Atlas/Bishwa/shell-identification-1-/src/labels/v1.00.mobile-net-label.json'
    with open(label_path, 'r') as f:
        label = json.load(f)
    label_dict_inv = {v:k for k,v in label.items()}
    return label_dict_inv


def predict(model, img):
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    predict = model_1.predict(image)
    
    label = np.argmax(predict)
    label_class = load_level()
    labelName = label_class[label]
    return labelName
    

if __name__ == '__main__':
    model = Mobile_net()
    model.load_weights('/home/atlas/Atlas/Bishwa/shell-identification-1-/models/v1.00.mobile-net')
    img_dir = '/home/atlas/Atlas/Bishwa/data/image'
    img = img_dir+'/Arthritica_helmsi_1_B.jpg'
    prediction = predict(model=model, img=img)
    print(prediction)