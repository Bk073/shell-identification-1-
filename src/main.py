from train import train, train_mobile_net
import evaluate

def main(train_model, *args):
    model_type = args[0] 
    if train_model:
        if model_type == 'base_model':
            train.train()
        if model_type == 'mobile_net':
            train_mobile_net.train()
    else:
        evaluate.eval()


if __name__ == '__main__':
    main(train_model=True, 'mobile_net')