#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from pickle import load
from keras.preprocessing.sequence import pad_sequences

application = Flask(__name__)

from keras.models import load_model
from keras.applications import ResNet50

incept_model = ResNet50(include_top=True)

from keras.models import Model

last = incept_model.layers[-2].output
modele = Model(inputs=incept_model.input, outputs=last)

model = load_model('./model_mi.h5')

new_dict = load(open('./vocab.pkl', 'rb'))
print(len(new_dict))
inv_dict = load(open('./vocab_idx.pkl', 'rb'))
print(len(inv_dict))


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print('step1')
    test_img = request.files['imagefile']
    if not test_img:
        return 'No pic uploaded!', 400

    print('step2')

    npimg = np.fromfile(test_img, np.uint8)
    test_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = np.reshape(test_img, (1, 224, 224, 3))

    print('step3')
    test_feature = modele.predict(test_img).reshape(1, 2048)

    # print(test_img, '========================')

    text_inp = ['startofseq']
    print('step5')
    count = 0
    MAX_LEN = 34
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(new_dict[i])

        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)

        prediction = np.argmax(model.predict([test_feature, encoded]))

        sampled_word = inv_dict[prediction]

        if sampled_word == 'endofseq':
            break
        caption = caption + ' ' + sampled_word

        text_inp.append(sampled_word)

    print('Done')

    return render_template('index.html', prediction_text='{}'.format(caption))


if __name__ == "__main__":
    application.run(debug=True)
