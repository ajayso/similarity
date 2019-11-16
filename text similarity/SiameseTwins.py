# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:43:24 2019

@author: Ajay Solanki
"""
# Imports
import keras
from keras import layers
from keras import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import pandas as pd
import matplotlib.pyplot as plt
from inputHandler import word_embed_meta_data,create_train_dev_set



class Comparator:

    def load_data(self):
    
        TRAIN_CSV = './data/train.csv'
        train_df = pd.read_csv(TRAIN_CSV)
        self.df = train_df
  
 
    
    def Execute_Model(self):
        EMBEDDING_DIM = 50
        MAX_SEQUENCE_LENGTH = 10
        RATE_DROP_LSTM = 0.17
        RATE_DROP_DENSE = 0.25
        NUMBER_LSTM = 50
        NUMBER_DENSE_UNITS = 50
        ACTIVATION_FUNCTION = 'relu'
        VALIDATION_SPLIT = 0.1
        
        sentences1 = list(self.df['question1'].astype(str))
        sentences2 = list(self.df['question2'].astype(str))
        is_similar = list(self.df['is_duplicate'])
        tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  EMBEDDING_DIM)

        embedding_meta_data = {
        	'tokenizer': tokenizer,
        	'embedding_matrix': embedding_matrix
        }
        sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
        nb_words = len(tokenizer.word_index) + 1
        embedding_layer = layers.Embedding(nb_words, siamese_config['EMBEDDING_DIM'], weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH, trainable=False)
        
       
        lstm_layer = layers.Bidirectional(layers.LSTM(NUMBER_LSTM, dropout=RATE_DROP_LSTM, recurrent_dropout=RATE_DROP_LSTM))

        sequence_1_input  = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        left_output = lstm_layer(embedded_sequences_1)
        
        sequence_2_input  = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        right_output = lstm_layer(embedded_sequences_2)
        
        merged = layers.concatenate([left_output, right_output], axis=-1)
        merged = BatchNormalization()(merged)
        merged = layers.Dropout(0.1)(merged)
        merged = layers.Dense(128, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        merged = layers.Dropout(0.1)(merged)
        predictions = layers.Dense(1, activation='sigmoid')(merged)
        
        model = Model([sequence_1_input, sequence_2_input], predictions)
        
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        model.summary()
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                               is_similar, MAX_SEQUENCE_LENGTH,
                                                                               VALIDATION_SPLIT)
        callbacks = [
                keras.callbacks.TensorBoard(
                log_dir='E:\workdirectory\Code Name Val Halen\DS Sup\DL\Chapter 15\logs',
                histogram_freq=1
                )
                ]
        
        self.history = model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=200, batch_size=64, shuffle=True,callbacks=callbacks)
       
        

    def plot(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        
        
comparator = Comparator()
comparator.load_data()
comparator.Execute_Model()
comparator.plot()