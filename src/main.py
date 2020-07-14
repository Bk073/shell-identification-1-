from train import train, train_mobile_net
import evaluate

def main(train_model, model_type): 
    if train_model:
        if model_type == 'base_model':
            train.train()
        if model_type == 'mobile_net':
            train_mobile_net.train()
    else:
        evaluate.eval()


if __name__ == '__main__':
    main(train_model=True, model_type='mobile_net')