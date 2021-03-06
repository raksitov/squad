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

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import json
#import yaml
import sys
import logging

import util

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json_data, generate_answers

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("h_learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("h_max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("h_dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("h_batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("h_hidden_size", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("h_context_len", 600, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("h_question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("h_embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")
tf.app.flags.DEFINE_integer("h_answer_len", 15, "The maximum answer length of your model")
tf.app.flags.DEFINE_integer("h_num_layers", 1, "The number of layers for RNN encoder")
tf.app.flags.DEFINE_string("h_cell_type", "gru", "The type of RNN cell.")
tf.app.flags.DEFINE_string("h_optimizer", "adam", "The type of optimizer.")
tf.app.flags.DEFINE_string("h_combiner", "concat", "Choose combiner for hidden states.")
tf.app.flags.DEFINE_integer("h_model_size", 75, "Size of the hidden states for the modelling layer")
tf.app.flags.DEFINE_integer("h_model_layers", 2, "The number of layers for modelling layer's RNN encoder")

# Addons
tf.app.flags.DEFINE_bool("prevent_end_before_start", True, "Prevents malformed spans from happening")
tf.app.flags.DEFINE_bool("multiply_probabilities", False, "Decide span based on the product of probabilites")
tf.app.flags.DEFINE_string("experiments_results", os.path.join(DEFAULT_DATA_DIR, 'experiments_results.json'), "Results of completed experiments")
tf.app.flags.DEFINE_bool("train_loss", False, "Adds train loss graph calculated over the whole train dataset")
tf.app.flags.DEFINE_bool("modeling_layer_uses_rnn", True, "Use RNN for modelling layer instead of FF")
tf.app.flags.DEFINE_bool("use_bidaf", False, "Whether to use basic attention or bidaf.")
tf.app.flags.DEFINE_bool("use_rnn_for_ends", False, "Whether to use rnn for predicting span ends.")
tf.app.flags.DEFINE_bool("share_encoder", True, "Whether to share weights for questions and answers encoding.")
tf.app.flags.DEFINE_bool("reuse_question_states", False, "Whether to reuse RNN states for questions when encoding answers.")
tf.app.flags.DEFINE_bool("ensemble", False, "Whether to evaluate ensemble of models.")

# Overrides for hparams
tf.app.flags.DEFINE_integer("batch_size", None, "Batch size override for eval")
tf.app.flags.DEFINE_integer("answer_len", None, "The maximum answer length override for eval")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print "Looking for model at %s..." % train_dir
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())

def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Print out Tensorflow version
    print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    # Define train_dir
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR,
        util.experiment_name(FLAGS.experiment_name))

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    def get_glove_fname(size):
      if size == 42:
        return "glove.42B.300d.txt"
      return "glove.6B.{}d.txt".format(size)
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR,
        get_glove_fname(FLAGS.h_embedding_size))

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.h_embedding_size)

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = os.path.join(FLAGS.data_dir, "train.context")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
    dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

    # Initialize model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train":

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:

            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)


    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)


    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        if not FLAGS.ensemble:
          with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict, _ = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
              f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
              print "Wrote predictions to %s" % FLAGS.json_out_path
        else:
          ckpts = FLAGS.ckpt_load_dir.split(',')
          data = []

          class AttrDict(dict):
            def __init__(self, *args, **kwargs):
              super(AttrDict, self).__init__(*args, **kwargs)
              self.__dict__ = self

          for idx, ckpt in enumerate(ckpts):
            print 'Currently processing model number {}'.format(idx)
            with open(os.path.join(ckpt, "flags.json")) as reader:
              #flags = yaml.safe_load(reader)
              flags_utf = json.load(reader)
            def from_utf(utf_string):
              try:
                return utf_string.encode('utf-8')
              except:
                return utf_string
            flags = {from_utf(key) : from_utf(flags_utf[key]) for key in flags_utf}
            flags = AttrDict(flags)
            flags.h_batch_size = FLAGS.batch_size
            flags.h_answer_len = FLAGS.answer_len
            tf.reset_default_graph()
            qa_model = QAModel(flags, id2word, word2id, emb_matrix)
            qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)
            with tf.Session(config=config) as sess:
              initialize_model(sess, qa_model, ckpt, expect_exists=True)
              answers_dict, confidence_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)
              data.append((answers_dict, confidence_dict))
          answers_dict = {}
          for uuid in data[0][0]:
            best = data[0][0][uuid]
            best_conf = data[0][1][uuid]
            best_dict = {best : best_conf}
            for ans, conf in data[1:]:
              best_dict[ans[uuid]] = best_dict.get(ans[uuid], 0.) + conf[uuid]
              if best_dict[ans[uuid]] > best_conf:
                best_conf = best_dict[ans[uuid]]
                best = ans[uuid]
            answers_dict[uuid] = best
          print "Writing predictions to %s..." % FLAGS.json_out_path
          with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
            print "Wrote predictions to %s" % FLAGS.json_out_path


    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
