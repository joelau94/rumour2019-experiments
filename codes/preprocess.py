import argparse

import data


def main():
  parser = argparse.ArgumentParser(description='Data preprocess')

  parser.add_argument('--dicts-file', type=str,
                      default='../data/dicts.pkl')

  parser.add_argument('--train-data-file', type=str,
                      default='../data/train.pkl')
  parser.add_argument('--dev-data-file', type=str,
                      default='../data/dev.pkl')
  parser.add_argument('--test-data-file', type=str,
                      default='../data/test.pkl')

  parser.add_argument('--raw-train-file', type=str,
                      default='../data/train.txt')
  parser.add_argument('--raw-dev-file', type=str,
                      default='../data/dev.txt')
  parser.add_argument('--raw-test-file', type=str,
                      default='../data/test.txt')
  parser.add_argument('--embed-pret-file', type=str,
                      default='../data/glove.6B.300d.txt')

  parser.add_argument('--min-freq', type=int, default=2)

  args = parser.parse_args()

  data.preprocess_train(args.raw_train_file,
                        args.dicts_file,
                        args.train_data_file,
                        args.embed_pret_file,
                        args.min_freq)
  data.preprocess_dev_test(args.raw_dev_file,
                           args.dicts_file,
                           args.dev_data_file)
  data.preprocess_dev_test(args.raw_test_file,
                           args.dicts_file,
                           args.test_data_file)


if __name__ == '__main__':
  main()
