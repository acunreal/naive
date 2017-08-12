#!/usr/bin/python
#coding:utf-8

import model as m
import json
import tensorflow as tf
import numpy as np

default_model_path='./ckpt/default_model.ckpt'
batch_size=m.batch_size

class NaiveSegmenter(object):
    def __init__(self, model_name=None,max_sentence_length=400):
        self.max_sentence_length=max_sentence_length
        with open('config.txt','r') as config_file:
            config=json.loads(config_file.read().decode('utf-8'))
        self.char_dict=config['c_mapping']
        self.label_dict=config['l_mapping']
        self.reverse_labels_mapping={}
        for k,v in self.label_dict.items():
            self.reverse_labels_mapping[v]=k
        
        #'building graph'
        self.graph=m.graph(max_sentence_length=max_sentence_length)
        with tf.variable_scope('mode'):
            self.graph.build_graph(is_train=False)
            saver=tf.train.Saver()
            self.sess=tf.Session()
            saver.restore(self.sess,default_model_path)

    def cut(self,text):
        #print 'begin to cut'
        inputs=text.decode('utf-8').split('\n')
        inputs=[input_text.replace('\r','') for input_text in inputs if input_text!=' ']
        outputs=[]
        while True:
            outputs+=self.handle_batch_file(inputs[0:batch_size])
            inputs=inputs[batch_size:]
            if len(inputs)==0:
                break
        return '\n'.join(outputs).replace('。',' 。 ')
            

    def handle_batch_file(self,batch):
        char_batch=[]
        sequence_lens=[]
        for sentence in batch:
            sentence_char_arrays=[]
            for i in range(0,self.max_sentence_length):
                char_array=[]
                for j in range(i,i+2):
                    if not (j<0 or j>=len(sentence)):
                        c=sentence[j]
                        if self.char_dict.has_key(c):
                            char_array.append(self.char_dict[c])
                        else:
                            char_array.append(self.char_dict['UNK'])
                    else:
                        char_array.append(self.char_dict['PAD'])
                sentence_char_arrays.append(char_array)
            char_batch.append(sentence_char_arrays)
            sequence_lens.append(len(sentence))

        if len(char_batch)<batch_size:
            for i in range(len(char_batch),batch_size):
                sequence_lens.append(0)
                char_batch.append([[self.char_dict['PAD'] for _ in range(0,2)] 
                                    for _ in range(0,self.max_sentence_length)])

        char_batch_array=np.array(char_batch)
        #print char_batch
        logits,transition_params=self.sess.run([self.graph.logits,self.graph.transition_params],
                            feed_dict={self.graph.char_examples_placeholder:char_batch_array,
                                    self.graph.sequece_length_batch:sequence_lens})        
        predicts=[]
        for i in range(0,batch_size):
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                                    logits[i], transition_params)
            predicts.append(viterbi_sequence)

        outputs=[]
        for i,sentence in enumerate(batch):
            tokens=[{'text':char} for char in sentence]
            for j,token in enumerate(tokens):
                if j<self.max_sentence_length:
                    token['predict']=self.reverse_labels_mapping[predicts[i][j]]
            outputs.append(m.concat_tokens(tokens,sentence_length=self.max_sentence_length,
                                        use_predict=True))
        return outputs




