# -*- coding: utf-8 -*-

#%%
#Set working directory
import os
cwd = "C:\\Users\\mayan\\Desktop\\Data\\NTU\\Study\\AI and Big Data\\Group Assignment\\Team-7 Project"
os.chdir(cwd)

#%%
#################### modules for app ############################
from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse
###############################################################
#################### modules for chatbot ############################
import pandas as pd
import numpy as np
import nltk
import string
#import json
import re
import tensorflow as tf
#import random
import spacy
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
from matplotlib import pyplot as plt
#################################################################
#%%
#################### Import csv data ############################
df = pd.read_csv("Banking Intent Data.csv")
df = df.iloc[:,0:4]
#################################################################

#inputs are what user may write
#classes are categories for each type of input
#targets are also categories but repeated for each input
#responses are possible responses to tell the user
#intent_doc is a dictionary with key as categories and values as possible reponses

inputs = df.iloc[:,1].tolist()
targets = df.iloc[:,2].tolist()
classes = df.iloc[:,2].unique().tolist()
responses = df.iloc[:,3].unique().tolist()
intent_doc = dict(zip(classes, responses))
#%%
#####################################Text Preprocessing############################

#cleaning using regular expressions
def preprocessing(line):
    line = [re.sub(r'[^a-zA-z.?!\']', ' ', i) for i in line]
    line = [re.sub(r'[ ]+', ' ', i) for i in line]
    return line

inputs = preprocessing(inputs)

#removing punctuation
def remove_punctuation(text):
    no_punct="".join([c for c in text if c not in string.punctuation])
    return no_punct

inputs=[remove_punctuation(x.lower()) for x in inputs]

#Tokenizing the sentence into words to perform lemmatization- will reduce the vocab_size
tokenizer_mid= RegexpTokenizer(r'\w+')
inputs=[tokenizer_mid.tokenize(x.lower()) for x in inputs]

#Instantiating the lemmatizer class from nltk
lemmatizer=WordNetLemmatizer()

#creating the Lemmatizer function
def word_lemmatizer(text):
    lem_text=[lemmatizer.lemmatize(i) for i in text]
    return lem_text


#applying lemmatization on input
inputs=[word_lemmatizer(x) for x in inputs]


#joing the token back to form a sentence
def join_back(a):
    text=" ".join(a)
    return text
inputs=[join_back(x) for x in inputs]
#%%
############################Keras tokenization########################################

def tokenize_data(input_list):
    #word tokenisation, anything not in vocabulary(oov: out of vocab) goes to unk
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    # we fit it using input_list
    tokenizer.fit_on_texts(input_list)
    #converts texts to numbers
    input_seq = tokenizer.texts_to_sequences(input_list)
    #makes sure that the array is of equal length, by using 0 to fill at the start
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')
    
    return tokenizer, input_seq
   

# Applying tokenization on input data
tokenizer, input_tensor = tokenize_data(inputs)


#using the targets list, 
#1: word: create a dictionary of targets , where each target will have a number, 
#2: categorical target: also create a list, which will be targets, but in numerical form based on the value assigned above
#3: categorical tensor: one hot encoded categorical target

def create_categorical_target(targets):
    word={}
    categorical_target=[]
    counter=0
    for trg in targets:
        if trg not in word:
            word[trg]=counter
            counter+=1
        categorical_target.append(word[trg])
    
    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v,k) for k, v in word.items())

# Creating one-hot encoded categorical target matrix for each category
target_tensor, trg_index_word = create_categorical_target(targets)

#for getting shape of the input and target matrices
print('input shape: {} and output shape: {}'.format(input_tensor.shape, target_tensor.shape))

#splitting our input and target matrxies into training and test set
from sklearn.model_selection import train_test_split

input_seq_train,input_seq_test,Y_train,Y_test=train_test_split(input_tensor,target_tensor,test_size=0.5)

#%%
#####################################Building the Model####################################################

# hyperparameters
epochs=30
vocab_size=len(tokenizer.word_index) + 1
embed_dim=128###learning very slow for higher dimensions.....
#units=128
target_length=Y_train.shape[1]


keras.backend.clear_session()####use this to erase the architecture of one model and reload the new one...

################################model Architecture 1- LSTM###############################

sequence_length=input_seq_train.shape[1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=sequence_length))
model.add(tf.keras.layers.SpatialDropout1D(0.2))
model.add(tf.keras.layers.LSTM(embed_dim,dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(target_length, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# train the model
history=model.fit(input_seq_train, Y_train, epochs=epochs, callbacks=[early_stop])

#Plots to see the training loss and accuracy
plt.title('Loss&Accuracy on train-set')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.legend()
plt.show();

#Evaluate the model
model.evaluate(input_seq_test,Y_test)


############################Model Architecture -2 Bidirectional LSTM#####################################

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_dim, dropout=0.2)),
    tf.keras.layers.Dense(embed_dim, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(target_length, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# train the model
history=model.fit(input_seq_train, Y_train, epochs=epochs, callbacks=[early_stop])

#Plots to see the training loss and accuracy
plt.title('Loss& Accuracy on train-set')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.legend()
plt.show();


#Evaluate the model
model.evaluate(input_seq_test,Y_test)

#%%
## function for predicting target label and return the response
def response(sentence):
    #convert to lower case
    sentence = sentence.lower()
    #pre-processing
    sentence = re.sub(r'[^a-zA-z.?!\']', ' ', sentence) 
    sentence = re.sub(r'[ ]+', ' ', sentence)
    #remove puncutations
    sentence=remove_punctuation(sentence)
    #break down sentence
    sentence = tokenizer_mid.tokenize(sentence)
    #lemmatization
    sentence=word_lemmatizer(sentence)
    sentence = join_back(sentence)

    sent_seq = []
    doc = nlp(repr(sentence))
    
    # split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # handle the unknown words error
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    # predict the category of input sentences
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)
    
    # choice a random response for predicted sentence
    return intent_doc[trg_index_word[pred_class[0]]]

# chat with bot
    '''
print("Note: Enter 'quit' to break the loop.")
while True:
    input_ = input('You: ')
    if input_.lower() == 'quit':
        break
    res = response(input_)
    print('Bot: {}'.format(res))
    print()
    '''
#%%
######################### APP ###################################
app = Flask(__name__)

@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()

    chatbot_text = response(incoming_msg)
    msg.body(chatbot_text)
        #responded=True
    
    return str(resp)


if __name__ == '__main__':
    app.run()
#%%
