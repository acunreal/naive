
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
import numpy as np

class MMForwardRnnCell(rnn_cell.RNNCell):

    def __init__(self, transition_params):
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        state = tf.expand_dims(state, 2)
        transition_scores = state + self._transition_params
        new_alphas = inputs + tf.reduce_max(transition_scores, [1])
        return tf.to_float(tf.squeeze(tf.argmax(transition_scores,1))), new_alphas

class MMBackwardRnnCell(rnn_cell.RNNCell):

    def __init__(self):
        pass

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        batch_size=inputs.get_shape()[0]
        num_tags=inputs.get_shape()[1]
        flattend_inputs=tf.reshape(inputs,[-1])
        flattened_state=tf.reshape(state,[-1])
        offsets=tf.range(batch_size,dtype=tf.int64)*num_tags
        new_state=tf.gather(flattend_inputs,flattened_state+offsets)
        new_state=tf.expand_dims(new_state,1)
        return new_state, new_state

def lengths_to_masks(lengths, max_length):

    tiled_ranges = tf.tile(
      tf.expand_dims(tf.range(max_length), 0),
      [tf.shape(lengths)[0], 1])
    lengths=tf.expand_dims(lengths, 1)
    masks=tf.to_float(
      tf.to_int64(tiled_ranges) < tf.to_int64(lengths))
    return masks


def max_margin(inputs,targets,sequence_lengths,is_train=True):
    (batch_size,max_sequence_len,num_tags)=(inputs.get_shape()[0].value,
                                        inputs.get_shape()[1].value,
                                        inputs.get_shape()[2].value)

    with tf.variable_scope('max_margin'):
        transition_matrix=tf.get_variable('transition000',[num_tags,num_tags],
                                                initializer=tf.random_uniform_initializer(-0.2,0.2))

        first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
        first_input = tf.squeeze(first_input, [1])
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        forward_cell = MMForwardRnnCell(transition_matrix)
        word_scores, alphas = rnn.dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths - 1,
            initial_state=first_input,
            dtype=tf.float32)


        masks=lengths_to_masks(sequence_lengths,max_sequence_len)

        masks_1=lengths_to_masks(sequence_lengths-1,max_sequence_len)
        reversed_masks_1=tf.abs(masks_1-1)

        padding_values=tf.tile(tf.expand_dims(tf.expand_dims(tf.argmax(alphas,1),1),2),[1,max_sequence_len,num_tags])

        trellis=tf.multiply(tf.concat([word_scores,tf.zeros([batch_size,1,num_tags])],1),
                                    tf.tile(tf.expand_dims(masks_1,2),[1,1,num_tags]))
        trellis=trellis+tf.multiply(tf.to_float(padding_values),
                                tf.tile(tf.expand_dims(reversed_masks_1,2),[1,1,num_tags]))
        trellis=tf.to_int64(trellis)

        backword_cell=MMBackwardRnnCell()

        predicts,_=rnn.dynamic_rnn(
            cell=backword_cell,
            inputs=tf.reverse(trellis,[1]),
            sequence_length=tf.constant(max_sequence_len,dtype=tf.int64,shape=[20]),
            initial_state=tf.expand_dims(tf.argmax(alphas,1),1))
        predicts=tf.to_int32(tf.squeeze(predicts))
        predicts=tf.reverse(predicts,[1])
        predict_score=tf.reduce_max(alphas,1)

        #debug
        #predicts=tf.to_int32(tf.argmax(tf.concat([tf.expand_dims(first_input,1),word_scores],1),2))

        if is_train:
            masks=lengths_to_masks(sequence_lengths,max_sequence_len)
            margin_loss=0.2*tf.reduce_sum(
                            tf.multiply(tf.cast(tf.not_equal(predicts,targets),tf.float32),masks),1)
            target_alone_score=tf.multiply(inputs,tf.one_hot(targets,num_tags,dtype=tf.float32))
            target_alone_score=tf.reduce_sum(target_alone_score,2)
            target_alone_score=tf.multiply(target_alone_score,masks)
            target_alone_score=tf.reduce_sum(tf.reshape(target_alone_score,[batch_size,max_sequence_len]),1)

            target_minus=tf.slice(targets,[0,0],[batch_size,max_sequence_len-1])
            target_plus=tf.slice(targets,[0,1],[batch_size,max_sequence_len-1])
            m_mat=tf.gather(transition_matrix,target_minus) #shape [batch_size,sentence_l-1,entity_count]
            flattened_m=tf.reshape(m_mat,[-1])
            flattened_tag_p=tf.reshape(target_plus,[-1])
            offsets=tf.range(batch_size*(max_sequence_len-1))*num_tags
            tags=flattened_tag_p+offsets
            target_transition_score=tf.gather(flattened_m,tags)
            target_transition_score=tf.reshape(target_transition_score,[-1,max_sequence_len-1])
            target_transition_score=tf.multiply(target_transition_score,tf.slice(masks,[0,1],[-1,-1,]))
            target_transition_score=tf.reduce_sum(target_transition_score,1)
            target_score=target_transition_score+target_alone_score
            #loss=tf.reduce_mean(tf.maximum(tf.zeros([batch_size]),predict_score+margin_loss-target_score))
            loss=tf.reduce_mean(predict_score+margin_loss-target_score)
        else:
            loss=None
            target_score=None
            target_alone_score=None
            target_transition_score=None
        return (loss,predicts,predict_score,transition_matrix,trellis,
            target_transition_score,target_alone_score,target_score)
       


