from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def params_dict():
  return tf.flags.FLAGS.__flags

def hparams_names():
  return [name for name in sorted(params_dict().keys()) if
      name.startswith('h_')]

def hparams_to_str():
  ordered_values = [str(params_dict()[name]) for name in hparams_names()]
  return ':'.join(ordered_values)

def convert(value):
  if value[0].isalpha():
    return str(value)
  if value.find('.') != -1:
    return float(value)
  return int(value)

def maybe_parse(name):
  values = name.split(':')
  if len(values) != len(hparams_names()):
    return False
  for idx, name in enumerate(hparams_names()):
    params_dict()[name] = convert(values[idx])
  # Overrides for eval.
  for param_to_override in ('batch_size', 'answer_len'):
    if params_dict()[param_to_override] is not None:
      params_dict()['h_{}'.format(param_to_override)] = params_dict()[param_to_override]
  return True

def experiment_name(experiment_name):
  if experiment_name:
    maybe_parse(experiment_name)
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
