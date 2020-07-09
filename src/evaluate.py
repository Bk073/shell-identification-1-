from models import base_model
from data.data_generator import test_data_generator, create_dataframe
from matplotlib import pyplot as plt


def eval_plot(loss, accuracy):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range, loss, label='Test Loss')
    plt.legend(loc='lower right')
    plt.title('Accuracy and loss')


def eval():
    test_data_gen = test_data_generator()
    base_model = base_model.BaseModel(test_data_gen=test_data_gen)
    base_model.load_model()
    loss, accuracy = base_model.eval()
    eval_plot(loss, accuracy)