'''
Created on 24 Aug 2018

@author: jwong
'''

import tensorflow as tf
import string

from utils.DataReader import MessagesReader
from utils.Processors import MessageProcessor, IndexMaps
from model.seq2seq import Seq2SeqModel

tf.enable_eager_execution()

class Config:
    MAX_MSGLEN = 60
    BATCH_SIZE = 32
    embedding_dim = 300    
    
if __name__ == '__main__':
    reader = MessagesReader()
    dat = reader.ReadAndProduceDataset()
    config = Config()
    print(len(dat))
    queryMap = IndexMaps([q for q,a in dat])
    responseMap = IndexMaps([a for q,a in dat])
    processor = MessageProcessor(config, queryMap, responseMap)
    dataset = processor.processDataset(dat, queryMap, responseMap)
    
    model = Seq2SeqModel(config, queryMap, responseMap)
    model.buildModel()
    model.train(dataset)
    f = open('jwonglog.txt', 'w')
    
    print('Awakening...')
    while True:
        msg = str(raw_input('input:'))
        if msg == 'exit':
            break
        inputTensor = processor.processOneSentence(msg)
        #print(model.predictMessage(inputTensor))
        result = model.predictMessage(inputTensor)
        result = filter(lambda x: x in set(string.printable), result)
        print(result)
    
    