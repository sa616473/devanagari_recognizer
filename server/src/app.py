from flask import Flask, render_template, request

import numpy as np
import pandas as pd

from tensorflow.keras.models import model_from_json
import  tensorflow as tf

from PIL import  Image

import re
import sys
import os
import base64

# import codecs
import cv2

sys.path.append(os.path.abspath('./model'))

# from load import *

#intit flask app

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



def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])

def predict():
    classes = pd.read_csv('./model/class_labels.csv')
    imgData = request.get_data()
    convertImage(imgData)
        
    im = Image.open('output.png')
    im = im.resize((32,32))
    im.save('resized.png')

    x = cv2.imread('resized.png')
    x = x / 255.
    x = x[np.newaxis, :, :, :1]

    model = init()

    out = model.predict(x)
    print(out)
    response = classes.iloc[(np.argmax(out))][1]
    print(response)
    return response



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8008))
    app.run(debug=True, port=port, host="0.0.0.0")