from flask import Flask, request
from flask_cors import CORS
import json
from models.mobile_net import Mobile_net
from predict import predict

app = Flask(__name__)

CORS(app)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("request files", request.files)
        file = request.files['file']
        model = Mobile_net()
        model.load_weights('/home/atlas/Atlas/Bishwa/shell-identification-1-/models/v1.01.mobile-net/')
        prediction = predict(model=model, img=img)
        print("predicted_image_label", prediction)
        return json.dumps({"label": prediction})

if __name__ == "__main__":
    app.run(debug=True)