from train import train, train_mobile_net
import evaluate

def main(train_model, model_type): 
    print(train_model)
    if train_model:
        if model_type == 'base_model':
            train.train()
        if model_type == 'mobile_net':
            train_mobile_net.train()
    else:
        evaluate.eval(model=model_type)

def eval_(model_type):
    evaluate.eval(model_type=model_type)


if __name__ == '__main__':
#     main(train_model=True, model_type='mobile_net')
    eval_(model_type='mobile_net')