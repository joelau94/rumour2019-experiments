import argparse
import cPickle as pkl
import os
import sys

import experiment


def main():
  parser = argparse.ArgumentParser(description='RumourEval Sampling')

  parser.add_argument('--config-file', type=str,
                      default='../models/config.pkl')
  parser.add_argument('--raw-data-file', type=str,
                      default='../data/train.txt')

  args = parser.parse_args()

  cfg = experiment.Config()
  cfg.load(args.config_file)
  exp = experiment.Experiment(cfg.config)

  exp.sample(args.raw_data_file)


if __name__ == '__main__':
  main()
