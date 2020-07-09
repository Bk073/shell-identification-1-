from train import train
import evaluate

def main(train_model):
    if train_model:
        train.train()
    else:
        evaluate.eval()


if __name__ == '__main__':
    main(train_model=True)