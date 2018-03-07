from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def hparams_to_str():
  hparams_names = [name for name in sorted(tf.flags.FLAGS.__flags.keys()) if
      name.startswith('h_')]
  ordered_values = [str(tf.flags.FLAGS.__flags[name]) for name in hparams_names]
  return ':'.join(ordered_values)

def experiment_name(experiment_name):
  if experiment_name:
    return experiment_name
  return hparams_to_str()
