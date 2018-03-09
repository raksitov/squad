import argparse
import os
import random


random.seed(42)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, required=True)
    parser.add_argument('--n_dev', type=int, required=True)
    parser.add_argument('--n_test', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = setup_args()

    data = {}    
    for name in ('context', 'question', 'span'):
        data[name] = [line for line in open(os.path.join(args.data_path,
          '{}.{}'.format('train', name)))]

    indices = range(len(data['span']))
    random.shuffle(indices)

    amounts = (0, args.n_train, args.n_dev, args.n_test)
    for key in data:
      for amounts_idx, split_type in enumerate(('train', 'dev', 'test')):
        with open(os.path.join(args.output_path,
            '{}.{}'.format(split_type, key)), 'w') as writer:
          for idx in xrange(sum(amounts[:amounts_idx + 1]),
              sum(amounts[:amounts_idx + 2])):
            writer.write(data[key][indices[idx]])


if __name__ == '__main__':
    main()
