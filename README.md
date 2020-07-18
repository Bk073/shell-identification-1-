# shell-identification

The aim of this project is to classify the sea shells into different species.
The dataset consists of 7894 shell species with 29622 samples, where totally 59244 shell images are present.

Each species has shell samples ranging from 1 to 87 respectively, and every shell samples has two photographs with frontal and lateral view.

Each image is labelled with its scientific name and of size 300*400 pixels.

# Requirements:
- pip install requirements.txt
 
- matplotlib==3.1.2
- numpy==1.18.0
- pandas==0.25.3
- scikit-learn==0.22.1
- scipy==1.4.1
- seaborn==0.9.0
- tensorflow-gpu==2.2.0
- keras

Before training and making prediction make sure all the path are set according to your setup.
# Train model:
Uncomment ` main(train_model=True, model_type='mobile_net')` in main.py to train the model
and uncomment `eval_(model_type='mobile_net')`to evaluate the model.

`python main.py`

# Predict:

`python predict.py` to predict stored images
`python app.py` to predict via server
