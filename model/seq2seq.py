'''
Created on 8 Sep 2018

@author: jwong
'''
import tensorflow as tf
import os
import numpy as np
from encoder import Encoder
from decoder import Decoder

class Seq2SeqModel(object):
    def __init__(self, config, queryMap, responseMap):
        self.queryMap = queryMap
        self.responseMap = responseMap
        self.embedding_dim = config.embedding_dim
        self.num_units = 1024
        self.batch_size = 32
        
    def buildModel(self):
        self.encoder = Encoder(len(self.queryMap.wordToIDmap), 
                          self.embedding_dim, 
                          self.num_units, 
                          self.batch_size)
        self.decoder = Decoder(len(self.responseMap.wordToIDmap),
                          self.embedding_dim, 
                          self.num_units, 
                          self.batch_size)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        
        self.checkpoint_dir = './training_checkpointsSat2'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.contrib.eager.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder,
                                         optimizer_step=tf.train.get_or_create_global_step())
    
    def _lossFunction(self, truth, pred):
        mask = 1 - np.equal(truth, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=truth, logits=pred) * mask
        return tf.reduce_mean(loss_)
        
    def train(self, dataset):
        num_epochs = 6
        
        for epoch in range(num_epochs):
            hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            
            for (batch, (x, lab)) in enumerate(dataset):
                loss = 0
                
                with tf.GradientTape() as tape:
                    #get encoder outputs and final state
                    enc_output, enc_hidden = self.encoder(x, hidden)
                    
                    #init state of decoder with final state of encoder
                    dec_hidden = enc_hidden
                    
                    #first input is batch of <begin> to get first pred
                    dec_input = tf.expand_dims(
                        [self.responseMap.wordToIDmap['<begin>']] * self.batch_size, 1)
                    
                    #Teacher forcing
                    for timeStep in range(1, lab.shape[1]):
                        preds, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                        
                        #computing loss for current timestep in each pred in batch
                        loss += self._lossFunction(lab[:, timeStep], preds)
                        
                        #set label of prev timestep as input to next
                        dec_input = tf.expand_dims(lab[:, timeStep], 1)
                
                batch_loss = (loss / int(lab.shape[1]))
                total_loss += batch_loss
                variables = self.encoder.variables + self.decoder.variables
                
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))
                #print('Batch: {}'.format(batch))
                
                if (batch % 5 == 0):
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
            
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss))
            
    def predictMessage(self, inputTensor):
        #if restorePath is not None:
        #    model = self.checkpoint.restore(restorePath)
            
        result = ''
        
        hidden = [tf.zeros((1, self.num_units))]
        enc_out, enc_hidden = self.encoder(inputTensor, hidden)
        
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.responseMap.wordToIDmap['<begin>']], 0)
        
        for timeStep in range(60):
            preds, dec_hidden,_ = self.decoder(
                dec_input, dec_hidden, enc_out)
            
            pred_id = tf.multinomial(preds, num_samples=1)[0][0].numpy()
            
            result += self.responseMap.idToWordmap[pred_id] + ' '
            
            if self.responseMap.idToWordmap[pred_id] == '<end>':
                return result.encode("utf-8")
            
            dec_input = tf.expand_dims([pred_id], 0)
        
        return result.encode("utf-8")
        
    def restoreModel(self):
        print('Restoring ' + self.checkpoint_dir)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

            
            
            
            
        
        
        