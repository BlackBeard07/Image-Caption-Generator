# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import pickle

import os
import pickle
import numpy as np
import tensorflow 
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.applications import ResNet50, DenseNet201
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,add


# loading the saved model
#loaded_model_resnet = pickle.load(open('C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Documents/Data_science/model_resnet.sav', 'rb'))
#loaded_model = pickle.load(open('C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Documents/Data_science/models.sav', 'rb'))
#loaded_pickle = pickle.load(open('C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Documents/Data_science/tokenizer.pickle', 'rb'))
with open('C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Documents/Data_science/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
loaded_model = tensorflow.keras.models.load_model('C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Documents/Data_science/models.h5')    
#resnet_model = tf.keras.models.load_model('C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Documents/Data_science/model_resnet_weights.h5')
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
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
    print( sentence )   

resnet_model = ResNet50()
# restructuring the model since we don't need the last softmax(prediction layer)
resnet_model = Model (inputs = resnet_model.inputs, outputs = resnet_model.layers[-2].output)




image_path = "C:/Users/mihir anand/OneDrive - Indian Institute of Technology Guwahati/Pictures/Photos/Untitled Export/IMG-3.jpg"#path of your image
# load image
image = load_img(image_path, target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# preprocess image for vgg
image = preprocess_input(image)
# extract features
feature = resnet_model.predict(image, verbose=0)
# predict from the trained model
caption = predict_caption(loaded_model, feature, tokenizer, 35)
final_caption = remove_start_end_tokens(caption)