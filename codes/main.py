import argparse
import cPickle as pkl
import os
import sys

import experiment


def main():
  parser = argparse.ArgumentParser(description='RumourEval')

  parser.add_argument('--test', dest='test', action='store_true')
  parser.add_argument('--config-file', type=str,
                      default='../models/config.pkl')

  parser.add_argument('--train-data-file', type=str,
                      default='../data/train.pkl')
  parser.add_argument('--dev-data-file', type=str,
                      default='../data/dev.pkl')
  parser.add_argument('--test-data-file', type=str,
                      default='../data/test.pkl')
  parser.add_argument('--dicts-file', type=str,
                      default='../data/dicts.pkl')
  parser.add_argument('--embed-pret-file', type=str,
                      default='../data/glove.6B.300d.txt')

  parser.add_argument('--keep-prob', type=float, default=0.9)
  parser.add_argument('--sdqc-weight', type=float, default=0.9)

  parser.add_argument('--embed-dim', type=int, default=300)
  parser.add_argument('--sent-hidden-dims', type=int, nargs='+',
                      default=[256, 256])
  parser.add_argument('--branch-hidden-dims', type=int, nargs='+',
                      default=[256, 256])
  parser.add_argument('--attn-dim', type=int, default=256)
  parser.add_argument('--sdqc-hidden-dim', type=int, default=512)
  parser.add_argument('--veracity-hidden-dim', type=int, default=512)

  parser.add_argument('--seed', type=int, default=23)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--beta1', type=float, default=0.9)
  parser.add_argument('--beta2', type=float, default=0.99)
  parser.add_argument('--eps', type=float, default=1e-8)
  parser.add_argument('--clip-norm', type=float, default=1.0)
  parser.add_argument('--global-norm', type=float, default=5.0)

  parser.add_argument('--ckpt', type=str,
                      default='../models/model')
  parser.add_argument('--max-ckpts', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--max-steps', type=int, default=1000000)
  parser.add_argument('--print-interval', type=int, default=50)
  parser.add_argument('--save-interval', type=int, default=1000)

  args = parser.parse_args()

  dicts = pkl.load(open(args.dicts_file, 'rb'))
  vocab_size = len(dicts['i2w'])

  cfg = experiment.Config()
  if args.test:
    cfg.load(args.config_file)
  else:
    cfg.config = {
        'train_data_file': args.train_data_file,
        'dev_data_file': args.dev_data_file,
        'test_data_file': args.test_data_file,
        'dicts_file': args.dicts_file,
        'embed_pret_file': args.embed_pret_file,

        'keep_prob': args.keep_prob,
        'sdqc_weight': args.sdqc_weight,

        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'sent_hidden_dims': args.sent_hidden_dims,
        'branch_hidden_dims': args.branch_hidden_dims,
        'attn_dim': args.attn_dim,
        'sdqc_hidden_dim': args.sdqc_hidden_dim,
        'veracity_hidden_dim': args.veracity_hidden_dim,

        'seed': args.seed,
        'lr': args.lr,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'eps': args.eps,
        'clip_norm': args.clip_norm,
        'global_norm': args.global_norm,

        'ckpt': args.ckpt,
        'max_ckpts': args.max_ckpts,
        'batch_size': args.batch_size,
        'max_steps': args.max_steps,
        'print_interval': args.print_interval,
        'save_interval': args.save_interval
    }

    if not os.path.isdir(os.path.dirname(args.config_file)):
      os.mkdir(os.path.dirname(args.config_file))
    cfg.save(args.config_file)

  exp = experiment.Experiment(cfg.config)

  if args.test:
    exp.test()
  else:
    exp.train()


if __name__ == '__main__':
  main()
