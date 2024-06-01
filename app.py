# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:40:20 2023

@author: mihir anand
"""

import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.applications import ResNet50, DenseNet201
import cv2
from PIL import Image,ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache_resource
def decorder_model():
    model = load_model('model.h5')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    return tokenizer

@st.cache_resource
def load_resnet():
    resnet_model = ResNet50()
    resnet_model = Model (inputs = resnet_model.inputs, outputs = resnet_model.layers[-2].output)
    return resnet_model

st.write("""
        # Image Caption Generator
        """)

#loading models and tokenizer
model = decorder_model()
resnet_model = load_resnet()
tokenizer = load_tokenizer()

file = st.file_uploader("Please upload an Image to generate caption", type = ['jpg', 'png', 'jpeg'])


def generate_features(model, image_data):
    target_size = (224, 224)
    image = ImageOps.fit(image_data, target_size, Image.ANTIALIAS)
    img_array = np.asarray(image)
    image_reshaped = img_array[np.newaxis,...]
    features = model.predict(image_reshaped)
    return features

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def pred_caption(img_feature, model, tokenizer):
    in_text = 'startseq'
    max_length = 35
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([img_feature, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text

def remove_start_end_tokens(raw_caption):
    words = raw_caption.split()
    words = words[1:-1]
    sentence = " ".join([ str(elm) for elm in words])
    return sentence

if file is None:
    st.text("No Image selected")
else:
    image = Image.open(file)
    #displaying image
    st.image(image, use_column_width = True)
    feature = generate_features(resnet_model, image)
    predicted_caption = pred_caption(feature, model, tokenizer)
    predicted_caption = remove_start_end_tokens(predicted_caption)
    st.success(predicted_caption)
    st.text("The caption generated might not be fully convincing but hey!! nobody is perfect:-")
