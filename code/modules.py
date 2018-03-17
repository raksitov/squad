# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob, num_layers=1, cell_type='gru',
        scope='RNNEncoder', combiner='concat'):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        def get_cell():
          if cell_type == 'gru':
            return rnn_cell.GRUCell(self.hidden_size)
          elif cell_type == 'lstm':
            return rnn_cell.LSTMCell(self.hidden_size)
          elif cell_type == 'layer_norm':
            return LayerNormBasicLSTMCell(self.hidden_size,
                dropout_keep_prob=keep_prob)
          else:
            raise Exception('Unknown cell type: {}'.format(cell_type))
        dropout = lambda: DropoutWrapper(get_cell(), input_keep_prob=self.keep_prob)
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.use_multi_layer_rnn = num_layers > 1
        self.rnn_cell_fw = dropout()
        if self.use_multi_layer_rnn:
          self.rnn_cell_fw = [dropout() for _ in xrange(num_layers)]
        self.rnn_cell_bw = dropout()
        if self.use_multi_layer_rnn:
          self.rnn_cell_bw = [dropout() for _ in xrange(num_layers)]
        self.scope = scope
        self.combiner = combiner

    def build_graph(self, inputs, masks=None):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(self.scope):
            input_lens = None
            if masks is not None:
              input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            if self.use_multi_layer_rnn:
              outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                  cells_fw=self.rnn_cell_fw,
                  cells_bw=self.rnn_cell_bw, 
                  inputs=inputs, 
                  sequence_length=input_lens, 
                  dtype=tf.float32)
              outputs = tf.split(outputs, 2, axis=2)
            else:
              # Note: fw_out and bw_out are the hidden states for every timestep.
              # Each is shape (batch_size, seq_len, hidden_size).
              outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw,
                  self.rnn_cell_bw, inputs, sequence_length=input_lens, dtype=tf.float32)

            if self.combiner == 'concat':
              # Concatenate the forward and backward hidden states
              outputs = tf.concat(outputs, 2)
            elif self.combiner == 'sum':
              outputs = outputs[0] + outputs[1]
            elif self.combiner == 'max':
              outputs = tf.maximum(outputs[0], outputs[1])
            elif self.combiner == 'mean':
              z = tf.constant([0.5])
              outputs = (outputs[0] + outputs[1]) * z
            else:
              raise Exception('Unknown combiner type: {}'.format(self.combiner))


            # Apply dropout
            outputs = tf.nn.dropout(outputs, self.keep_prob)

            return outputs


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, scope='BasicAttn'):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
        """
        self.keep_prob = keep_prob
        self.scope = scope

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope(self.scope):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

class BiDAF(object):
    """Module for BiDAF.

    """

    def __init__(self, keep_prob):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
        """
        self.keep_prob = keep_prob

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values and vice versa.
        For each key and value, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size).
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          keys_output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the keys attention output; the weighted sum of the values
            (using the keys attention distribution as weights).
          values_output: Tensor shape (batch_size, num_keys).
            This is the values attention output; the weighted sum of the values
            (using the values attention distribution as weights).
        """
        with vs.variable_scope("BiDAF"):

            # Calculate attention distribution
            w1 = tf.get_variable('w1', shape=(keys.shape[2]))
            w2 = tf.get_variable('w2', shape=(keys.shape[2]))
            w3 = tf.get_variable('w3', shape=(keys.shape[2]))

            w1 = tf.nn.dropout(w1, self.keep_prob)
            w2 = tf.nn.dropout(w2, self.keep_prob)
            w3 = tf.nn.dropout(w3, self.keep_prob)

            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = (tf.matmul(keys * w1, values_t) + # shape (batch_size, num_keys, num_values)
                tf.expand_dims(tf.reduce_sum(keys * w2, axis=2), 2) +
                tf.expand_dims(tf.reduce_sum(values * w3, axis=2), 1))

            attn_logits_values_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, keys_attn_dist = masked_softmax(attn_logits, attn_logits_values_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            keys_output = tf.matmul(keys_attn_dist, values) # shape (batch_size, num_keys, value_vec_size)
            keys_output = tf.nn.dropout(keys_output, self.keep_prob)

            m = tf.reduce_max(attn_logits, axis=2) # shape (batch_size, num_keys)
            _, values_attn_dist = masked_softmax(m, keys_mask, 1)
            values_attn_dist = tf.expand_dims(values_attn_dist, 1)

            values_output = tf.matmul(values_attn_dist, keys)
            values_output = tf.nn.dropout(values_output, self.keep_prob)


            return keys_output, values_output
