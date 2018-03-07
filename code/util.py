from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def params_dict():
  return tf.flags.FLAGS.__flags

def hparams_to_str():
  hparams_names = [name for name in sorted(params_dict().keys()) if
      name.startswith('h_')]
  ordered_values = [str(params_dict()[name]) for name in hparams_names]
  return ':'.join(ordered_values)

def experiment_name(experiment_name):
  if experiment_name:
    return experiment_name
  return hparams_to_str()

def serialize_results(best_dev_f1, best_dev_em, num_epochs):
  return {
      'best_dev_f1' : best_dev_f1,
      'best_dev_em' : best_dev_em,
      'num_epochs' : num_epochs,
      'hparams' : {name : params_dict()[name] for name
        in params_dict().keys() if name.startswith('h_')},
      }
