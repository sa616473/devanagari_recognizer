from flask import Flask, render_template, request

import numpy as np
import pandas as pd

from tensorflow.keras.models import model_from_json
import  tensorflow as tf
from tensorflow.keras.preprocessing import image

from werkzeug.utils import secure_filename

import sys
import os


sys.path.append(os.path.abspath('./model'))



app = Flask(__name__)




def init():
    json_file = open('./model/model_128_64.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("./model/model_128_64.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam', metrics=['accuracy'])

    return loaded_model

global model

model = init()

global classes
classes = pd.read_csv('./model/class_labels.csv')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def model_predict():
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    print(file_path)

    img = image.load_img(file_path, target_size=(32,32))
    x = image.img_to_array(img)
    x = x / 255.
    x = x[np.newaxis, :, :, :1]

    out = model.predict(x)
    print(out)
    response = classes.iloc[(np.argmax(out))][1]
    print(response)
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)