from collections import Counter, defaultdict
import cPickle as pkl
import itertools
import random

import numpy as np

import utils

import pdb


class Dataset(object):
  """Dataset.
  records format:
  list of threads [
    list of tweets[
      word_ids,
      word_pret_ids,
      sdqc_label
    ],
    veracity_label
  ]

  """

  def __init__(self, datafile, shuffle=False, seed=None):
    self.records = pkl.load(open(datafile, 'rb'))
    self.shuffle = shuffle

    random.seed(seed)
    self.order = list(range(len(self.records)))
    if self.shuffle:
      random.shuffle(self.order)
    self.cursor = 0

    self.has_pret = False
    if len(self.records[0][0][0]) == 3:
      self.has_pret = True

    utils.print_log('{} records loaded.'.format(len(self.records)))

  def get_next(self, n=1):
    if n > len(self.records):
      utils.print_log('Batch size must be smaller than dataset size.')
      exit(0)
    start = self.cursor
    stop = (self.cursor + n) % len(self.records)
    if stop > start:
      batch = [self.records[i] for i in self.order[start:stop]]
      self.cursor = stop
    else:
      batch = [self.records[i] for i in self.order[start:]]
      random.shuffle(self.order)
      self.cursor = 0

    sent_length = []
    branch_length = []
    xs = []
    xs_pret = []
    ys_sdqc = []
    ys_veracity = []
    for thread in batch:
      branch_length.append(len(thread[0]))
      ys_veracity.append(thread[1])
      slen = []
      x = []
      x_pret = []
      y = []
      for tweet in thread[0]:
        slen.append(len(tweet[0]))
        x.append(tweet[0])
        if self.has_pret:
          x_pret.append(tweet[0])
        y.append(tweet[-1])
      sent_length.append(slen)
      xs.append(x)
      if self.has_pret:
        xs_pret.append(x_pret)
      ys_sdqc.append(y)

    # (batch,)
    max_sent_length = max(itertools.chain(*sent_length))
    max_branch_length = max(branch_length)

    # (batch, thread_len)
    sent_length = [l + [0] * (max_sent_length - len(l))
                   for l in sent_length]

    # (batch, thread_len, sent_len)
    xs = [[tweet + [0] * (max_sent_length - len(tweet))
           for tweet in thread] +
          [[0] * max_sent_length] * (max_branch_length - len(thread))
          for thread in xs]

    X_pret = None
    if self.has_pret:
      xs_pret = [[tweet + [0] * (max_sent_length - len(tweet))
                  for tweet in thread] +
                 [[0] * max_sent_length] * (max_branch_length - len(thread))
                 for thread in xs_pret]
      X_pret = np.array(xs, dtype=np.int64)

    # (batch, thread_len)
    ys_sdqc = [[tweet for tweet in thread] +
               [0] * (max_branch_length - len(thread))
               for thread in ys_sdqc]

    X = np.array(xs, dtype=np.int64)
    Y_sdqc = np.array(ys_sdqc, dtype=np.int64)
    Y_veracity = np.array(ys_veracity, dtype=np.int64)
    sent_length = np.array(sent_length, dtype=np.int64)
    branch_length = np.array(branch_length, dtype=np.int64)

    return X, X_pret, Y_sdqc, Y_veracity, sent_length, branch_length


def parse_txt(txtfile):
  """Text data format:
  original tweet 1 ||| support ||| true
  reply tweet 1 ||| deny
  reply tweet 2 ||| query
  <newline>
  original tweet 2 ||| support  ||| false
  reply tweet 3 ||| deny
  reply tweet 4 ||| comment
  <newline>
  ...
  """

  raw = open(txtfile, 'r').read()

  def parse_orig(l):
    x, y, v = l.strip().split('|||')
    return v.strip(), [x.strip().split(), y.strip()]

  def parse_reply(l):
    x, y = l.strip().split('|||')
    return [x.strip().split(), y.strip()]

  def parse_thread(t):
    t = t.strip().split('\n')
    v, orig = parse_orig(t[0])
    replies = []
    for reply in t[1:]:
      replies.append(parse_reply(reply))
    return [[orig] + replies, v]

  if raw.strip() == '':
    return []

  records = [parse_thread(thread) for thread in raw.strip().split('\n\n')]

  return records


def build_dicts(records, output_file, embed_pret_file=None, min_freq=2):
  word_bucket = []
  for thread in records:
    for tweet in thread[0]:
      word_bucket += tweet[0]

  word_count = Counter(word_bucket)

  dicts = {}

  dicts['i2w'] = ['_UNK_'] + \
      [k for k, v in word_count.iteritems() if v >= min_freq]
  dicts['w2i'] = defaultdict(int)
  dicts['w2i'].update({w: i for i, w in enumerate(dicts['i2w'])})
  utils.print_log('Vocab Size: {}'.format(len(dicts['i2w'])))

  dicts['i2t'] = ['support', 'deny', 'query', 'comment']
  dicts['t2i'] = {w: i for i, w in enumerate(dicts['i2t'])}

  dicts['i2v'] = ['true', 'false', 'unverified']
  dicts['v2i'] = {w: i for i, w in enumerate(dicts['i2v'])}

  if embed_pret_file:
    dicts['i2w_pret'] = ['_UNK_'] + \
        [line.strip().split()[0]
         for line in open(embed_pret_file, 'r')
         if line.strip() != '']
    dicts['w2i_pret'] = defaultdict(int)
    dicts['w2i_pret'].update({
        w: i for i, w in enumerate(dicts['i2w_pret'])})
    utils.print_log('Pretrained Vocab Size: {}'.format(len(dicts['i2w_pret'])))

  pkl.dump(dicts, open(output_file, 'wb'))

  return dicts


def index_records(raw_records, dicts):
  records = []
  for thread in raw_records:
    thread_ids = []
    veracity_label = dicts['v2i'][thread[1]]
    for tweet in thread[0]:
      tweet_ids = list(map(lambda w: dicts['w2i'][w], tweet[0]))
      sdqc_label = dicts['t2i'][tweet[1]]
      if 'w2i_pret' in dicts:
        tweet_ids_pret = list(map(lambda w: dicts['w2i_pret'][w], tweet[0]))
        thread_ids.append([tweet_ids, tweet_ids_pret, sdqc_label])
      else:
        thread_ids.append([tweet_ids, sdqc_label])
    records.append([thread_ids, veracity_label])

  return records


def preprocess_train(raw_data_file, dicts_file, datafile,
                     embed_pret_file=None, min_freq=2):

  raw_records = parse_txt(raw_data_file)
  if len(raw_records) < 1:
    utils.print_log('No records found in {} !'.format(datafile))
    return

  dicts = build_dicts(raw_records, dicts_file, embed_pret_file, min_freq)

  records = index_records(raw_records, dicts)
  utils.print_log('Data: {}\n  {} threads, {} tweets.'.format(
      datafile, len(records), sum(map(len, records))
  ))

  pkl.dump(records, open(datafile, 'wb'))


def preprocess_dev_test(raw_data_file, dicts_file, datafile):
  raw_records = parse_txt(raw_data_file)
  if len(raw_records) < 1:
    utils.print_log('No records found in {} !'.format(datafile))
    return

  dicts = pkl.load(open(dicts_file, 'rb'))

  records = index_records(raw_records, dicts)
  utils.print_log('Data: {}\n  {} threads, {} tweets.'.format(
      datafile, len(records), sum(map(len, records))
  ))

  pkl.dump(records, open(datafile, 'wb'))
