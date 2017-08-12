#!/usr/bin/python
#coding:utf-8


from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import json
import os
import random

import numpy as np
import tensorflow as tf
import word2vec
import diy_layer

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
tf.logging.set_verbosity(tf.logging.WARN)

#model parameters
batch_size=20   #20 for train
char_embedding_dimension=512
left_window=0
right_window=1
window_size=left_window+right_window+1
max_sentence_length=100
test_sentence_length=650
lr=0.2
lstm_input_dimension=char_embedding_dimension*window_size
hidden_size=150
keep_rate=0.8
crf_layer=True
epoch=501


#data_parameters
train_eval_set_cut_line=90000
set_name='msr'
using_bakeoff_2015_score=True

raw_data_filename='./data/msr_training.utf8'
char_vec_file='./vecs/char_embedding@%s'%(char_embedding_dimension)+'-'+set_name+'.bin'
char_vec_file=char_vec_file.replace('.utf8','.bin')
vec_train_file='./vecs/embedding_corpus'+'-'+set_name+'.txt'
test_set_file='./data/msr_test.utf8'

#placeholder and default
entity_count=5
char_dictionary_count=None
char_counter={}



def generate_ribao_raw_data_file():
    top_dir='./2014/'
    data=''
    for folder_name in os.listdir(top_dir):  
        if 'store' in folder_name.lower():
            continue
        second_dir=top_dir+folder_name
        for filename in os.listdir(second_dir):
            if filename.lower()[0]=='c' and '-' in filename.lower() and '.txt' in filename.lower():
                
                with open(second_dir+'/'+filename,'r') as content_file:
                    print second_dir+'/'+filename
                    content=content_file.read().decode('utf-8').replace(']',' ]')
                    content=content.replace('\r','').replace(r'。','\n').split('\n')
                    content_obj=[[word.split('/')[0] for word in sent.replace('\r','').split(' ')] 
                                                    for sent in content if len(sent)>2]
                    file_data=('||').join([' '.join(sent) for sent in content_obj]).replace('[','').replace(']','')
                    file_data=file_data.replace('|| ||','\n').replace('||','\n').replace('\n ”','”')
                    file_data=file_data.replace('   ',' ').replace('  ',' ').replace('  ',' ')
                    #print file_data
                    if len(file_data)>2:
                        data=data+file_data
        with open(raw_data_filename,'a') as raw_data_file:
            data=data.replace('  ',' ')
            raw_data_file.write(data)
            data=''


def train_char2vec():
    if not os.path.exists(char_vec_file):
        if not os.path.exists(vec_train_file):
            with open(raw_data_filename,'r') as pku_train_file:
                print 'reading raw data...'
                corpus=pku_train_file.read().decode('utf-8')
                print 'splitting corpus'
                corpus=corpus.split('\n')
                total_length=len(corpus)
                for i,sent in enumerate(corpus):
                    if i%1000==0:
                        print 'concat train file: %s'%(i/float(total_length)*100)+'%'
                    with open(vec_train_file,'a') as vt_file:
                        vt_file.write((' ').join(sent.replace('\r','').replace(' ',''))+'\n')
        word2vec.word2vec(vec_train_file,char_vec_file,size=char_embedding_dimension,verbose=True)
    model=word2vec.load(char_vec_file)
    return model

def json_print(obj):
    print json.dumps(obj,ensure_ascii=False,indent=2)

def create_utts_from_file(filename,
                        enforce_sample=False,
                        sentence_length=max_sentence_length,
                        splitter=' ',
                        shuffle_data=False,
                        test_data=False):
    utts=[]
    with open(filename,'r') as data_file:
        content=data_file.read().decode('utf-8')
        corpus=content.split('\n')
        if enforce_sample:
            from collections import Counter
            counts=Counter(content.replace('\n',splitter).replace('\r','').split(splitter))
            most_common=counts.most_common(1000)
            least_words= [k for k,v in counts.most_common() if v<5]
            multiplers={}
            for word in least_words:
                includes_counts=0
                for k,_ in most_common:
                    if k==' ' or k=='':
                        continue
                    if word.find(k)!=-1 and k!=word:
                        includes_counts+=counts[k]

                mul_count=int(includes_counts/counts[word]/10)-1
                if mul_count>0:
                    import math
                    multiplers[word]=int(math.log(mul_count,10))
            multiplers_array=[]
            for k,v in multiplers.items():
                for x in range(v):
                    if len(multiplers_array)<2:
                        multiplers_array.append(k)
                        continue
                    index=random.randint(0,len(multiplers_array)-1)
                    multiplers_array.insert(index,k)
            multiplers_string='  。  '.join(multiplers_array)
            corpus.insert(int(len(corpus)/8.0),multiplers_string)

        total_length=len(corpus)
        for i,sent in enumerate(corpus):
            if i%10000==0 and i>0:
                print 'processing %s to utts: %s/%s'%(filename,i,total_length)
                if i==750000:
                    break
            tokens=[]
            if not test_data:
                words=sent.replace('\r','').split(splitter)
                for word in words:
                    if len(word)==1:
                        tokens.append({'text':word,'label':'S'})
                        continue
                    for i,c in enumerate(word):
                        if i==0:
                            tokens.append({'text':c,'label':'B'})
                        elif i==len(word)-1:
                            tokens.append({'text':c,'label':'E'})
                        else:
                            tokens.append({'text':c,'label':'M'})
                utts.append(tokens)
            else:
                sent=sent.replace('\r')
                for c in sent:
                    tokens.append({'text':c})
        new_utts=[]
        for tokens in utts:
            while len(tokens)>sentence_length:
                i=sentence_length-1
                while tokens[i]['text'] not in ['，','。','？','！','：','、','；'] and i>0:
                    i-=1
                if i==0:
                    break
                new_utts.append(tokens[:i+1])
                tokens=tokens[i+1:]
            new_utts.append(tokens[:sentence_length])
        if shuffle_data:
            random.shuffle(new_utts)
        print 'total data line length is %s'%(len(new_utts))
    return new_utts


def prepare(model,utts):
    #讀取數據，並計算漢字和詞組個數
    char_dictionary_index=0
    labels_dictionary_index=0
    chars_mapping={}
    labels_mapping={}
    reverse_labels_mapping={}
    for tokens in utts:
        for token in tokens:
            if not chars_mapping.has_key(token['text']):
                chars_mapping[token['text']]=char_dictionary_index
                char_counter[token['text']]=0
                char_dictionary_index+=1
            char_counter[token['text']]+=1
            if not labels_mapping.has_key(token['label']):
                labels_mapping[token['label']]=labels_dictionary_index
                reverse_labels_mapping[labels_dictionary_index]=token['label']
                labels_dictionary_index+=1
    
    if crf_layer:
        labels_mapping['PAD']=labels_dictionary_index
        reverse_labels_mapping[labels_dictionary_index]='PAD'
    
    unknown_set=set()
    reindex=0
    new_chars_mapping={}
    char_vecs=[]
    for k,v in char_counter.items():
        if v<=3 or k==u' ':
            unknown_set.add(k)
        else:
            try:
                char_vecs.append(model[k].tolist())
                new_chars_mapping[k]=reindex
                reindex+=1
            except Exception:
                unknown_set.add(k)


    new_chars_mapping['UNK']=reindex
    new_chars_mapping['PAD']=reindex+1
    for _ in range(0,2):
        char_vecs.append([0.0 for _ in range(0,char_embedding_dimension)])
    return new_chars_mapping,labels_mapping,reverse_labels_mapping,char_vecs


class graph(object):
    """docstring for graph"""
    cell_fw=None
    cell_bw=None
    def __init__(self,max_sentence_length=max_sentence_length):
        self.sentence_length=max_sentence_length

    def build_graph(self,char_vecs=None,is_train=True):

        #cnn for char feature extraction
        self.char_examples_placeholder=tf.placeholder(tf.int32,shape=[batch_size,self.sentence_length,window_size]) 
        if not char_vecs:
            self.char_emb=tf.get_variable('char_emb',
                                    shape=np.array([5561,512]),
                                    initializer=tf.constant_initializer(0.0,dtype=tf.float32))
        else:
            self.char_emb=tf.get_variable('char_emb',
                                        shape=np.array(char_vecs).shape,
                                        initializer=tf.constant_initializer(np.array(char_vecs),dtype=tf.float32))
        char_example_embeddings=tf.nn.embedding_lookup(self.char_emb,self.char_examples_placeholder)
        shape_list=char_example_embeddings.get_shape().as_list()
        sentence_length=shape_list[1]


        self.input_tensor=tf.reshape(char_example_embeddings,[batch_size,sentence_length,-1])

        if not graph.cell_fw:
            graph.cell_fw=tf.contrib.rnn.LSTMCell(hidden_size)
            graph.cell_bw=tf.contrib.rnn.LSTMCell(hidden_size)

        if is_train:
            cell_fw=tf.contrib.rnn.DropoutWrapper(graph.cell_fw, output_keep_prob=keep_rate)
            cell_bw=tf.contrib.rnn.DropoutWrapper(graph.cell_bw, output_keep_prob=keep_rate)
            
        else:
            cell_fw=graph.cell_fw
            cell_bw=graph.cell_bw

        

        self.sequece_length_batch=tf.placeholder(tf.int32,[batch_size])
        (rnn_fw,rnn_bw),states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,
                                                    self.input_tensor,
                                                    sequence_length=self.sequece_length_batch,
                                                    dtype=tf.float32)



        self.rnn_output=tf.concat([rnn_fw,rnn_bw],2)
        self.softmax_w=tf.get_variable('softmax_w',
                                    [entity_count,hidden_size*2],
                                    initializer=tf.random_uniform_initializer(-0.2,0.2))
        self.softmax_b=tf.get_variable('softmax_b',
                                    [entity_count],
                                   initializer=tf.constant_initializer(0.0,dtype=tf.float32))
        self.rnn_matrix=tf.reshape(self.rnn_output,[-1,hidden_size*2])
        logits=tf.matmul(self.rnn_matrix,self.softmax_w,transpose_b=True)+self.softmax_b
        self.logits=tf.reshape(logits, [batch_size, self.sentence_length, entity_count])
        self.targets=tf.placeholder(tf.int32,[batch_size,self.sentence_length])

        

        if crf_layer:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, 
                                        self.targets,self.sequece_length_batch)
            loss = tf.reduce_mean(-log_likelihood)
            
        else:
            (loss,self.predicts,self.predict_score,self.trans_p,self.trellis,self.tts,
                self.tas,self.ts)=diy_layer.max_margin(self.logits,self.targets,
                                                            self.sequece_length_batch,is_train=is_train)

        if is_train:
            self.tvars = tf.trainable_variables()
            if crf_layer:
                self.lr = tf.Variable(lr, trainable=False)
                lossL2=tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if '_b' not in v.name])*10**(-4)*0.5
                self.cost=loss+lossL2
                optimizer=tf.train.AdagradOptimizer(self.lr,0.02)
            else:
                self.lr = tf.Variable(0.1, trainable=False)
                lossL2=tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if '_b' not in v.name])*10**(-3)*0.5
                optimizer=tf.train.AdagradOptimizer(self.lr)
                
            self.cost=loss+lossL2
            self.train_op=optimizer.minimize(self.cost)
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self.lr_update = tf.assign(self.lr, self.new_lr)



def batch(utts,char_dict,label_dict,test_data=False,max_sentence_length=100, max_epoch=1):
    
    epoch=0
    utt_i=0
    char_batch=[]
    labels_batch=[]
    tokens_batch=[]
    cache=[]
    while True:
        tokens=utts[utt_i]
        if len(cache)>utts:
            sentence_char_arrays=cache[utt_i]['sca']
            sentence_label_array=cache[utt_i]['sla']
        else:
            sentence_char_arrays=[]
            sentence_label_array=[]
            for i in range(0,max_sentence_length):
                if i<len(tokens):
                    token=tokens[i]
                    if not test_data:
                        sentence_label_array.append(label_dict[token['label']])
                else:
                    if crf_layer:
                        sentence_label_array.append(label_dict['PAD'])
                    else:
                        sentence_label_array.append(label_dict['S'])
                char_array=[]
                for j in range(i-left_window,i+right_window+1):
                    if not (j<0 or j>=len(tokens)):
                        c=tokens[j]['text']
                        if char_dict.has_key(c):
                            char_array.append(char_dict[c])
                        else:
                            char_array.append(char_dict['UNK'])
                    else:
                        char_array.append(char_dict['PAD'])
                sentence_char_arrays.append(char_array)
        if len(cache)<=utt_i:
            cache.append({'sca':sentence_char_arrays,'sla':sentence_label_array})
        char_batch.append(sentence_char_arrays)
        labels_batch.append(sentence_label_array)
        tokens_batch.append(tokens)
        if len(labels_batch)>=batch_size:
            yield np.array(char_batch),np.array(labels_batch),tokens_batch,epoch
            char_batch=[]
            labels_batch=[]
            tokens_batch=[]
        if utt_i+1<len(utts):
            utt_i+=1
        else:
            utt_i=0
            epoch+=1
            if epoch==max_epoch:
                if len(tokens_batch)>0:
                    for i in range(len(tokens_batch),batch_size):
                        char_batch.append([[char_dict['PAD'] for _ in range(0,window_size)] 
                                                        for _ in range(0,max_sentence_length) ])
                        if crf_layer:
                            labels_batch.append([[label_dict['PAD'] for _ in range(0,max_sentence_length)]])
                        else:
                            labels_batch.append([[label_dict['S'] for _ in range(0,max_sentence_length)]])
                yield None,None,None,None

def concat_tokens(tokens,sentence_length=max_sentence_length,use_predict=False):
    retstr=''
    if use_predict:
        key='predict'
    else:
        key='label'
    for token in tokens[0:sentence_length]:
        if token[key] in ['B','S']:
            retstr+=' '
        elif token[key]=='PAD':
            retstr+='[PAD]'
        retstr+=token['text']
    return retstr

def counting_tokens(tokens):
    recall=len([ token for i,token in enumerate(tokens) 
                    if token['label'] in ['B','S'] and i<max_sentence_length ])
    precision=len([ token for i,token in enumerate(tokens) 
                    if i<max_sentence_length and token['predict'] in ['B','S']])
    correct=0
    correct_opening=False
    for token in tokens:
        if token.has_key('predict'):
            if token['label']==token['predict']=='S':
                correct+=1

            if token['label']==token['predict']=='B':
                correct_opening=True

            if token['label']==token['predict']=='E' and correct_opening:
                correct_opening=False
                correct+=1
    return recall,precision,correct


def run_test(utts,graph,char_dict,label_dict,reverse_labels_mapping,
            sentence_length=max_sentence_length,out_put_file=False):
    if out_put_file:
        test_data=True
    else:
        test_data=False
    batch_reader=batch(utts,char_dict,label_dict,max_sentence_length=sentence_length,test_data=test_data)

    output_string=''
    precision_total=0
    recall_total=0
    correct_entity_count=0

    while True:
        char_examples,label_examples,tokens_batch,epoch=batch_reader.next()
        if epoch==None:
            break
        predicts=[]
        if crf_layer:
            logits,transition_params=sess.run([
                                        graph.logits,
                                        graph.transition_params],
                            feed_dict={graph.char_examples_placeholder:char_examples,
                                graph.sequece_length_batch:np.array([len(tokens) for tokens in tokens_batch])})
        
            
            for i in range(0,batch_size):
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                                        logits[i], transition_params)
                predicts.append(viterbi_sequence)

        else:
            predicts=sess.run(graph.predicts,
                feed_dict={graph.char_examples_placeholder:char_examples,
                    graph.sequece_length_batch:np.array([len(tokens) for tokens in tokens_batch])})
  
        for i,tokens in enumerate(tokens_batch):
            for j,token in enumerate(tokens):
                if j<sentence_length:
                    token['predict']=reverse_labels_mapping[predicts[i][j]]
            if out_put_file:
                output_string+=concat_tokens(tokens,
                                        sentence_length=test_sentence_length,
                                        use_predict=True)+'\n'

        for tokens in tokens_batch:
            r,p,c=counting_tokens(tokens)
            precision_total+=p
            recall_total+=r
            correct_entity_count+=c

    if not out_put_file:
        if correct_entity_count>0:
            print 'precision is %s,recall is %s'%(correct_entity_count/float(precision_total),
                                        correct_entity_count/float(recall_total))
        
    else:            
        with open(out_put_file,'wb') as output_file:
            output_file.write(output_string)



def score_output(filename):
    result_path='./result/'
    output_path='./icwb2-data/icwb2-data/scripts/'
    gold_path='./icwb2-data/icwb2-data/gold/'

    filepath=result_path+filename
    filename_main=filename.split('.')[0]
    score_file_path=output_path+filename_main+'_score.txt'

    with open(filepath,'r') as test_file:
        content=test_file.read().replace('。',' 。 ')

        with open(output_path+filename.replace('.txt','.utf8'),'wb') as utf8_file:
            utf8_file.write(content.encode('utf-8'))

    os.system(output_path+'score '+gold_path+set_name+'_training_words.utf8 '
            +gold_path+set_name+'_test_gold.utf8 '+output_path+filename_main+'.utf8 >'
            +score_file_path)

    with open(score_file_path,'r') as score_file:
        result_lines=score_file.read().split('\n')
        print '\n'.join(result_lines[-14:-1])



if __name__=='__main__':
    #generate_ribao_raw_data_file()
    print 'loading vec model'
    model=train_char2vec()
    print 'create training data'
    utts=create_utts_from_file(raw_data_filename,shuffle_data=False,enforce_sample=True,splitter='  ')
    eval_set=utts[train_eval_set_cut_line:]
    if test_set_file:
        test_set=create_utts_from_file(test_set_file,sentence_length=test_sentence_length,splitter='  ')
    else:
        test_set=None
    print 'preparing...'
    char_dict,label_dict,reverse_labels_mapping,char_vecs=prepare(model,utts)
    char_dictionary_count,entity_count=len(char_dict),len(label_dict)
    train_reader=batch(utts[:train_eval_set_cut_line],char_dict,label_dict,max_epoch=epoch)

    print 'building graph'
    train_graph=graph()
    test_graph=graph(max_sentence_length=test_sentence_length)
    with tf.variable_scope('mode') as scope:
        train_graph.build_graph(char_vecs)
        scope.reuse_variables()
        test_graph.build_graph(char_vecs,is_train=False)
    saver=tf.train.Saver()

    with tf.Session() as sess:
        #read ckpt if exist
        savepath='./ckpt/%s-model%s-500.ckpt.index'%(set_name,char_embedding_dimension)
        if os.path.exists(savepath):
            print 'loading prevous ckpt'
            saver.restore(sess,savepath.replace('.index',''))
        else:
            sess.run(tf.global_variables_initializer())

        count=0
        epoch_steps=0
        last_epoch=-1
        step_cost=0
        epoch_cost=0
        while  True:
            char_examples,label_examples,tokens_batch,epoch=train_reader.next()
            for tokens in [sent for sent in tokens_batch if len(sent)>max_sentence_length]:
                json_print(tokens)
            if epoch==None:
                savepath=saver.save(sess,'./ckpt/%s-model%s-final.ckpt'%(set_name,char_embedding_dimension))
                print 'model saved to file: %s'%(savepath)
                break
            if epoch!=last_epoch:
                last_epoch=epoch
                if epoch>0:
                    savepath=saver.save(sess,'./ckpt/%s-model%s-%s.ckpt'%(set_name,char_embedding_dimension,epoch))
                    print 'model saved to file: %s'%(savepath)
                    print 'epoch:%s,average_cost:%s'%(epoch,epoch_cost/float(epoch_steps))
                    epoch_steps=0
                    epoch_cost=0
                    output_file_name='%s_test@d%s-e%s.txt'%(set_name,char_embedding_dimension,epoch)
                    if test_set:
                        run_test(test_set,test_graph,
                            char_dict,
                            label_dict,
                            reverse_labels_mapping,
                            sentence_length=test_sentence_length,
                            out_put_file='./result/'+output_file_name)
                        print 'test file generated (%s,%s)'%(char_embedding_dimension,epoch)
                        if using_bakeoff_2015_score:
                            score_output(output_file_name)
                print '======'+set_name+'@%s'%(epoch)+'======'
                #sess.run(train_graph.lr_update,feed_dict={train_graph.new_lr:lr*lr_decay**epoch})


            cost,_,lr=sess.run([train_graph.cost,
                                train_graph.train_op,
                                train_graph.lr],
                    feed_dict={
                        train_graph.char_examples_placeholder:char_examples,
                        train_graph.targets:label_examples,
                        train_graph.sequece_length_batch:np.array([len(tokens) for tokens in tokens_batch]),
                        train_graph.new_lr:1.0
                    }
                )



            count+=1
            epoch_steps+=1
            step_cost+=cost
            epoch_cost+=cost
            if count%100==0:
                print 'steps:%s,average_cost:%s'%(count,step_cost/100)
                step_cost=0
            if count%1000==0:
                run_test(eval_set,test_graph,
                        char_dict,
                        label_dict,
                        reverse_labels_mapping,
                        sentence_length=test_sentence_length)








