'''
Created on 6 Sep 2018

@author: jwong
'''

import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.embeddingLayer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.num_units = num_units
        self.lstm = tf.keras.layers.GRU(num_units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')
                                         #return_sequences=True, #return all lstm node outputs
                                         #return_state=True, #also return the final node state
                                         #recurrent_activation=tf.sigmoid,
                                         #recurrent_initializer='glorot_uniform')
        '''
        Encoder(len(self.queryMap.wordToIDmap), 
                          self.embedding_dim, 
                          self.num_units, 
                          self.batch_size)
        '''
    def call(self, x, hidden):
        x = self.embeddingLayer(x)
        output, state = self.lstm(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.num_units))