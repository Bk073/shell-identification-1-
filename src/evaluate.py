from models import base_model
from data.data_generator import test_data_generator, create_dataframe
from matplotlib import pyplot as plt
import pandas as pd
from models.mobile_net import Mobile_net
import tensorflow as tf
from predict import load_level

def eval_plot(loss, accuracy):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range, loss, label='Test Loss')
    plt.legend(loc='lower right')
    plt.title('Accuracy and loss')


def eval(model):
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    img_dir = '/home/atlas/Atlas/Bishwa/data/image'
    test_df = pd.read_csv('/home/atlas/Atlas/Bishwa/shell-identification-1-/src/data/test_data.csv')
    test_data_gen = test_data_generator(test_df, IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH, img_dir=img_dir)
    if model_type == 'base_model':
        base_model = base_model.BaseModel(test_data_gen=test_data_gen)
        base_model.load_model()
        loss, accuracy = base_model.eval()
        eval_plot(loss, accuracy)
        
    if model_type == 'mobile_net':
        model = Mobile_net()
        model.load_weights('/home/atlas/Atlas/Bishwa/shell-identification-1-/models/v1.01.mobile-net/')
        predict = model.predict(test_data_gen)
        predict_indices = tf.math.argmax(predict, 1)
        
        label_dict_inv = load_level()

        prediction = [] 

        for index in predict_indices:
            prediction.append(label_dict_inv[index.numpy()])
        
        print(accuracy_score(prediction, test_df['label']))
        print(classification_report(test_df['label'], prediction))