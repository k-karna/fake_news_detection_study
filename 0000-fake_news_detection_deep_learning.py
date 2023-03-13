#!/usr/bin/env python
# coding: utf-8

# # Fake News Detection using Deep Learning Methods

# * Dataset: LIAR,LIAR Pants-On-Fire
# * Deep Learning Methods: ANN, RNN, CNN, GRU, LSTM, Bi-Directional LSTM

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.__version__)


# ### Importing Dataset

# LIAR dataset is divided into three set: 1. train set 2.test set 3. validation set. We will import each separately and as column do not have names to it, we will add for clearer understanding

# In[2]:


#Importing training set and adding column label names

train_data = pd.read_csv('train.tsv',sep = '\t',
                         names = ['File Type', 'Label', 'Statement', 'Context', 'Speaker',
                                  'Position', 'State', 'Party', 'n1','n2','n3','n4','n5','Source'])


# In[3]:


#checking training set

train_data.head()


# #### Removing all other redundant columns except 'Label' and 'Statement' as they will not required for fake news detection task 

# In[4]:


td= train_data.filter(['Label', 'Statement'],axis=1)
td


# #### Removing labels with HALF_TRUE 

# Because half-true is equal to half-false, and therfore label with HALF-TRUE cannot substantially contribute to news classification into true and false

# In[5]:


td1 = td[td.Label != 'half-true']
td1


# ### Changing Labels into 0 and 1

# * Labels with "TRUE", "MOSTLY TRUE" as True or 1
# * Labels with "FALSE", "BARELY-TRUE", "PANTS-FIRE" as Fake or 0

# In[6]:


mapping = {'true': 1, 'mostly-true': 1, 'false': 0, 'pants-fire': 0, 'barely-true': 0}
training_set =td1.replace({'Label': mapping})
training_set


# ## Test set and validation need to pre-processed as similar to Train Set

# * Additional columns needs to be removed
# * "Half-True" labels needs to be removed
# * Labels needs to re-labelled as 0 and 1 for fake and real news respectively

# ### Pre-processing on test set

# In[7]:


#Importing test set and adding column label names
test_data = pd.read_csv('test.tsv', sep = '\t',
                       names = ['File Type', 'Label', 'Statement', 'Context', 'Speaker',
                                  'Position', 'State', 'Party', 'n1','n2','n3','n4','n5','Source'])

test_data.head()


# In[8]:


# Removing redundant columns

test_set1= test_data.filter(['Label', 'Statement'],axis=1)
test_set1


# In[9]:


#Removing news with label as "HALF-TRUE"

tset = test_set1[test_set1.Label != 'half-true']
tset


# In[10]:


#chaning labels to 0 and 1 for fake and real news

mapping = {'true': 1, 'mostly-true': 1, 'false': 0, 'pants-fire': 0, 'barely-true': 0}
test_set =tset.replace({'Label': mapping})


# In[11]:


test_set


# ### Pre-processing on Validation set

# In[12]:


#Importing test set and adding column label names

valid_data = pd.read_csv('test.tsv', sep = '\t',
                       names = ['File Type', 'Label', 'Statement', 'Context', 'Speaker',
                                  'Position', 'State', 'Party', 'n1','n2','n3','n4','n5','Source'])
valid_data.head()


# In[13]:


#Removing all other labels expect statement and label

vd= valid_data.filter(['Label', 'Statement'],axis=1)
vd


# In[14]:


#removing news label with "HAF-LF_TRUE" label

vd1 = vd[vd.Label != 'half-true']
vd1


# In[15]:


#Changing news label into 0 and 1 for fake and real news
mapping = {'true': 1, 'mostly-true': 1, 'false': 0, 'pants-fire': 0, 'barely-true': 0}
valid_set = vd1.replace({'Label': mapping})


# In[16]:


valid_set


# # Natural Language Processing with the Text Data of News

# * Remove Punctuation
# * Remove Stopwords
# * Then, we will implement tokenizer to create padding sequences. Also, to calculate num_words, we will concatenate all news statement to gather exact number

# ### Removing Punctuation

# In[17]:


import string

def remove_punc(text):
    table = str.maketrans("","",string.punctuation)
    return text.translate(table)


# In[18]:


#removing punctuation from the news statement from the training set, test set, and validation set

training_set['Statement']=training_set['Statement'].map(lambda x: remove_punc(x))
test_set['Statement']=test_set['Statement'].map(lambda x: remove_punc(x))
valid_set['Statement']=valid_set['Statement'].map(lambda x: remove_punc(x))


# ### Removing stopwords

# In[19]:


#importing stopwords from the NLTK to remove ENGLISH stopwords, and convert further into lower within 
#remove_Stopwords function defined

import nltk
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))

def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


# In[20]:


#removing ENGLISH stopwords from the news statement from the training set, test set, and validation set

training_set['Statement']=training_set['Statement'].map(remove_stopwords)
test_set['Statement']=test_set['Statement'].map(remove_stopwords)
valid_set['Statement']=valid_set['Statement'].map(remove_stopwords)


# In[21]:


#Training set after removing punctuation, stopwords, and lowercasing all the words in the corpus
training_set['Statement']


# In[22]:


#Test set after removing punctuation, stopwords, and lowercasing all the words in the corpus
test_set['Statement']


# In[23]:


#Validation set after removing punctuation, stopwords, and lowercasing all the words in the corpus
valid_set['Statement']


# ### In the next step, we will be using TOKENIZER, and tokenizer requires num_words parameter

# * To find num_words, we need to the know total count of unique words in the train set, test set, and validation set
# * Therefore, we will be using counter collection of PYTHON. It counts the unique words

# #### Creating new set to concatnate all three sets to get exact no. of words in dataset

# In[24]:


new_set = pd.concat([training_set, test_set, valid_set], axis=0)
new_set


# In[25]:


from collections import Counter

def counter(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] +=1
    return count


# In[26]:


text = new_set['Statement']
total_count = counter(text)


# In[27]:


len(total_count)


# #### Creating parameters

# In[28]:


num_words = len(total_count)
oov_tok = "<oov>"
max_length = 20 #because its says average statement length is 17.9
trunc_type = 'post'
padding_type = 'post'


# #### Creating training set, training label, test set, test label, validation set to be used in neural network

# In[29]:


train_sentences = training_set['Statement']
train_labels = training_set['Label']


# In[30]:


test_sentences = test_set['Statement']
test_labels = test_set['Label']


# In[31]:


valid_sentences = valid_set['Statement']
valid_labels = valid_set['Label']


# #### Importing TOKENIZER

# In[32]:


from keras.preprocessing.text import Tokenizer


# In[33]:


#we will use all the words in the train set, test set, and validation set to tokenier
new_set_sentences = new_set['Statement']


# In[34]:


tokenizer = Tokenizer(num_words = num_words, oov_token = oov_tok)
tokenizer.fit_on_texts(new_set_sentences)


# In[35]:


#Creating word_index for each words tokenized

word_index = tokenizer.word_index
word_index


# In[36]:


train_sequences = tokenizer.texts_to_sequences(train_sentences)


# In[37]:


train_sequences[0]


# In[38]:


from keras.preprocessing.sequence import pad_sequences


# In[39]:


train_padded = pad_sequences(
                train_sequences, maxlen =max_length, padding=padding_type,truncating = trunc_type)


# In[40]:


train_padded[0]


# In[41]:


test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
                test_sequences, maxlen=20, padding="post",
                truncating = "post")


# In[42]:


valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
valid_padded = pad_sequences(valid_sequences, maxlen= 20, padding ="post",truncating="post")


# In[43]:


print(f"Shape of the train {train_padded.shape}")
print(f"shape of the test {test_padded.shape}")


# # Implementing DEEP LEARNING methods

# We will use keras to implement deep learning methods. Methods used are

# * Long Short Term Memory (LSTM)
# * Artificial Neural Network
# * Convolutional Neural Network (CNN)
# * Gated Recurrent Unit (GRU)
# * Bi-Directional LSTM
# * Recurrent Neural Network (RNN)

# ### Modelling LSTM

# In[44]:


#Importing relevant libraries to create desired neural networks

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam


# In[45]:


model = Sequential()


# #### Number of hidden layer, dropout, learning rate has been tried and tested many times over for better result

# In[46]:


model.add(Embedding(num_words, 32, input_length=max_length))
model.add(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2))
model.add(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2))
model.add(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2))
model.add(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2))
model.add(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2))
model.add(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2))
model.add(LSTM(32, activation = "relu", dropout = 0.1))
model.add(Dense(1, activation='sigmoid'))
opt = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])


# In[47]:


model.summary()


# #### Epochs and batch_size has been tested many times over for better results

# In[48]:


model.fit(train_padded, train_labels, epochs=30, batch_size=32, validation_data=(valid_padded, valid_labels))


# #### Changing float value in output to 0 and 1 with 0.5 as classification point

# In[50]:


predictions = model.predict(test_padded)
predictions = (predictions > 0.5)


# In[51]:


predictions.astype('int64')


# #### We will be focusing on Accuracy Score and Recall for fake news i.e 0 label for evaluating our model performance

# In[52]:


#Importing confusion matrix and accuracy score from SCIKIT-LEARN to understand results

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels,predictions))


# In[53]:


print("Accuracy Score for LSTM:",accuracy_score(test_labels,predictions))


# ## Modelling ANN

# In[54]:


from keras.layers import Flatten


# In[55]:


ann = Sequential()

#First input and hidden layer
ann.add(Embedding(num_words,32, input_length=max_length))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=32, activation='relu'))
ann.add(Flatten())
#Output layer
ann.add(Dense(units=1))

#compiling
opt = Adam(learning_rate=0.001)
ann.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[56]:


ann.summary()


# #### Epochs and batch_size has been tested many times over for better results

# In[57]:


#training ann
ann.fit(train_padded, train_labels, batch_size=32, validation_data=(valid_padded, valid_labels),epochs=40)


# In[58]:


predictions_ann = ann.predict(test_padded)
predictions_ann = (predictions_ann > 0.5)
predictions_ann.astype('int64')


# In[63]:


print(confusion_matrix(test_labels, predictions_ann))
print(classification_report(test_labels,predictions_ann))
print("Accuracy Score for ANN :",accuracy_score(test_labels,predictions_ann))


# ### Modelling CNN 1D

# In[124]:


from keras.layers import Conv1D, GlobalAveragePooling1D,Dense


# In[125]:


cnn = Sequential()

#First input and hidden layer
cnn.add(Embedding(num_words, 32, input_length=max_length))
cnn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
cnn.add(GlobalAveragePooling1D())
cnn.add(Dense(24,activation='relu'))
cnn.add(Dense(24,activation='relu'))
#Output layer
cnn.add(Dense(1, activation='sigmoid'))

#compiling
opt = Adam(learning_rate=0.00001)
cnn.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[126]:


cnn.summary()


# #### Epochs and batch_size has been tested many times over for better results

# In[141]:


#training cnn
cnn.fit(train_padded, train_labels, epochs=40, batch_size=32, validation_data=(valid_padded, valid_labels))


# In[142]:


predictions_cnn = cnn.predict(test_padded)
predictions_cnn = (predictions_cnn > 0.5)
predictions_cnn.astype('int64')


# In[143]:


print(confusion_matrix(test_labels, predictions_cnn))
print(classification_report(test_labels,predictions_cnn))
print("\n")
print("Accuracy Score for CNN:",accuracy_score(test_labels,predictions_cnn))


# ### Modelling GRU

# In[133]:


from keras.layers import GRU


# In[134]:


gru = Sequential()


# In[135]:


gru.add(Embedding(num_words, 32, input_length=max_length))
gru.add(GRU(32, return_sequences=True, activation="relu", recurrent_activation='sigmoid', dropout = 0.2))
gru.add(GRU(32, return_sequences=True, activation="relu", recurrent_activation='sigmoid', dropout = 0.2))
gru.add(GRU(32, return_sequences=True, activation="relu", recurrent_activation='sigmoid', dropout = 0.2))
gru.add(GRU(32, return_sequences=True, activation="relu", recurrent_activation='sigmoid', dropout = 0.2))
gru.add(GRU(32, return_sequences=True, activation="relu", recurrent_activation='sigmoid', dropout = 0.2))
gru.add(GRU(32, return_sequences=True, activation="relu", recurrent_activation='sigmoid', dropout = 0.2))
gru.add(GRU(32, activation = "relu", dropout = 0.1))
gru.add(Dense(1, activation='sigmoid'))
opt = Adam(learning_rate=0.0001)
gru.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])


# In[136]:


#printing GRU Model Summary
gru.summary()


# #### Epochs and batch_size has been tested many times over for better results

# In[137]:


gru.fit(train_padded, train_labels, epochs=25, batch_size=32, validation_data=(valid_padded, valid_labels))


# In[138]:


predictions_gru = gru.predict(test_padded)
predictions_gru = (predictions_gru > 0.50)
predictions_gru.astype('int64')


# In[140]:


print(confusion_matrix(test_labels, predictions_gru))
print(classification_report(test_labels,predictions_gru))
print("\n")
print("Accuracy Score for GRU :",accuracy_score(test_labels,predictions_gru))


# ### Modelling Bi-Directional LSTM

# In[78]:


from keras.layers import Bidirectional


# In[79]:


bilstm = Sequential()


# In[80]:


bilstm.add(Embedding(num_words, 32, input_length=max_length))
bilstm.add(Bidirectional(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2)))
bilstm.add(Bidirectional(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2)))
bilstm.add(Bidirectional(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2)))
bilstm.add(Bidirectional(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2)))
bilstm.add(Bidirectional(LSTM(32, return_sequences=True, activation="relu", dropout = 0.2)))
bilstm.add(Bidirectional(LSTM(32, activation = "relu", dropout = 0.2)))
bilstm.add(Dense(1, activation='sigmoid'))
opt = Adam(learning_rate=0.0001)
bilstm.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])


# In[81]:


bilstm.summary()


# #### Epochs and batch_size has been tested many times over for better results

# In[82]:


bilstm.fit(train_padded, train_labels, epochs=20, batch_size=32, validation_data=(valid_padded, valid_labels))


# In[83]:


predictions_bilstm = bilstm.predict(test_padded)
predictions_bilstm = (predictions_bilstm > 0.5)
predictions_bilstm.astype('int64')


# In[84]:


from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
print(confusion_matrix(test_labels, predictions_bilstm))
print(classification_report(test_labels,predictions_bilstm))


# In[85]:


print("Accuracy Score for Bi-Directional LSTM :",accuracy_score(test_labels,predictions_bilstm))


# ### Modelling RNN for Fake News Detection

# In[100]:


from keras.layers import SimpleRNN
from keras.layers import Flatten


# In[101]:


rnn = Sequential()

#First input and hidden layer
rnn.add(Embedding(num_words,32, input_length=max_length))
rnn.add(Dense(units=6, activation='relu'))
rnn.add(Dense(units=6, activation='relu'))
rnn.add(Dense(units=6, activation='relu'))
rnn.add(Dense(units=6, activation='relu'))
rnn.add(Dense(units=6, activation='relu'))
rnn.add(Flatten())
#Output layer
rnn.add(Dense(units=1))

#compiling
opt = Adam(learning_rate=0.0001)
rnn.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])


# #### RNN Model Summary

# In[102]:


rnn.summary()


# #### Epochs and batch_size has been tested many times over for better results

# In[106]:


rnn.fit(train_padded, train_labels, epochs=45, batch_size=32, validation_data=(valid_padded, valid_labels))


# In[107]:


predictions_rnn = rnn.predict(test_padded)
predictions_rnn = (predictions_rnn > 0.5)
predictions_rnn.astype('int64')


# In[108]:


print(confusion_matrix(test_labels, predictions_rnn))
print(classification_report(test_labels,predictions_rnn))
print("Accuracy Score for RNN :",accuracy_score(test_labels,predictions_rnn))


# In[ ]:




