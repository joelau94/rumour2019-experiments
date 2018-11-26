import cPickle as pkl
import os

import tensorflow as tf
import numpy as np

import data
import models
import utils


class Config(object):
  """Config"""

  def __init__(self):
    self.config = {
        'train_data_file': '../data/train.pkl',
        'dev_data_file': '../data/dev.pkl',
        'test_data_file': '../data/test.pkl',
        'embed_pret_file': '../data/glove.6B.300d.txt',
        'dicts_file': '../dicts.pkl',

        'keep_prob': 0.9,
        'sdqc_weight': 0.5,

        'vocab_size': 30000,
        'embed_dim': 300,
        'sent_hidden_dims': [256, 256],
        'branch_hidden_dims': [256, 256],
        'attn_dim': 256,

        'seed': 23,
        'lr': 0.1,
        'beta1': 0.9,
        'beta2': 0.99,
        'eps': 1e-8,
        'clip_norm': 1.0,
        'global_norm': 5.0,

        'ckpt': '../models/model',
        'max_ckpts': 20,
        'batch_size': 64,
        'max_steps': 1000000,
        'gemb_steps': 1000000,
        'print_interval': 50,
        'save_interval': 1000
    }

  def save(self, filename):
    pkl.dump(self.config, open(filename, 'wb'))

  def load(self, filename):
    self.config = pkl.load(open(filename, 'rb'))


class Experiment(object):
  """Experiment"""

  def __init__(self, config):
    self.config = config

  def train(self):
    train_data = data.Dataset(self.config['train_data_file'],
                              shuffle=True,
                              seed=self.config['seed'])
    dev_data = data.Dataset(self.config['dev_data_file'],
                            shuffle=False)

    train_graph = tf.Graph()

    with tf.Session(graph=train_graph) as sess:
      model = models.RumourDetectModel(
          embed_dim=self.config['embed_dim'],
          vocab_size=self.config['vocab_size'],
          sent_hidden_dims=self.config['sent_hidden_dims'],
          branch_hidden_dims=self.config['branch_hidden_dims'],
          attn_dim=self.config['attn_dim'],
          embed_pret_file=self.config['embed_pret_file'],
          dicts_file=self.config['dicts_file'],
          keep_prob=self.config['keep_prob'],
          reuse=None)

      model(is_train=True)
      loss = self.config['sdqc_weight'] * model.sdqc_loss + \
          (1. - self.config['sdqc_weight']) * model.veracity_loss

      optimizer = tf.train.AdamOptimizer(
          learning_rate=self.config['lr'],
          beta1=self.config['beta1'],
          beta2=self.config['beta2'],
          epsilon=self.config['eps'])
      gradients = optimizer.compute_gradients(loss)
      gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gradients if grad is not None]

      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      train_op = optimizer.apply_gradients(gradients,
                                           global_step=self.global_step)

      sess.run(tf.global_variables_initializer())
      if self.config['embed_pret_file']:
        model.embedder.init_pretrained_emb(sess)

      train_saver = tf.train.Saver(max_to_keep=self.config['max_ckpts'])
      ckpt_dir = os.path.dirname(self.config['ckpt'])
      if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
      ckpt = tf.train.latest_checkpoint(ckpt_dir)
      if ckpt:
        train_saver.restore(sess, ckpt)

      utils.print_log('Training ...')
      global_steps = sess.run(self.global_step)
      while global_steps <= self.config['max_steps']:

        X, X_pret, Y_sdqc, Y_veracity, sent_length, branch_length = \
            train_data.get_next(self.config['batch_size'])

        _, step_loss, global_steps = sess.run(
            [train_op, loss, self.global_step],
            feed_dict={
                model.word_ids: X,
                model.word_ids_pret: X_pret,
                model.sdqc_labels: Y_sdqc,
                model.veracity_labels: Y_veracity,
                model.sent_length: sent_length,
                model.branch_length: branch_length,
            })

        if global_steps % self.config['print_interval'] == 0:
          utils.print_log('Step {}: Loss = {}'.format(global_steps, step_loss))

        if global_steps % self.config['save_interval'] == 0:
          train_saver.save(sess, self.config['ckpt'],
                           global_step=global_steps)

          # validation
          batch_num = int(math.floor(len(dev_data.records) /
                                     self.config['batch_size']))
          sdqc_corr, sdqc_total, veracity_corr, veracity_total = 0, 0, 0, 0
          for _ in range(batch_num):
            X, X_pret, Y_sdqc, Y_veracity, sent_length, branch_length = \
                dev_data.get_next(batch_num)
            c1, t1, c2, t2 = sess.run(
                [model.sdqc_correct_count, model.sdqc_total_count,
                 model.veracity_correct_count, model.veracity_total_count],
                feed_dict={
                    model.word_ids: X,
                    model.word_ids_pret: X_pret,
                    model.sdqc_labels: Y_sdqc,
                    model.veracity_labels: Y_veracity,
                    model.sent_length: sent_length,
                    model.branch_length: branch_length,
                })
            sdqc_corr += c1
            sdqc_total += t1
            veracity_corr += c2
            veracity_total += t2

          utils.print_log('Step {}: SDQC Task Acc = {}, Veracity Task Acc = {}'
                          .format(global_steps,
                                  float(sdqc_corr) / sdqc_total,
                                  float(veracity_corr) / veracity_total))

  def test(self):
    test_data = data.Dataset(self.config['test_data_file'], shuffle=False)
    test_graph = tf.Graph()

    with tf.Session(graph=test_graph) as sess:
      model = models.RumourDetectModel(
          embed_dim=self.config['embed_dim'],
          vocab_size=self.config['vocab_size'],
          sent_hidden_dims=self.config['sent_hidden_dims'],
          branch_hidden_dims=self.config['branch_hidden_dims'],
          attn_dim=self.config['attn_dim'],
          embed_pret_file=self.config['embed_pret_file'],
          dicts_file=self.config['dicts_file'],
          keep_prob=1.0,
          reuse=None)

      model(is_train=False)
      sess.run(tf.global_variables_initializer())
      if self.config['embed_pret_file']:
        model.embedder.init_pretrained_emb(sess)

      saver = tf.train.Saver(max_to_keep=self.config['max_ckpts'])
      ckpt_dir = os.path.dirname(self.config['ckpt'])
      ckpt = tf.train.latest_checkpoint(ckpt_dir)
      saver.restore(sess, ckpt)

      utils.print_log('Testing ...')
      batch_num = int(math.floor(len(test_data.records) /
                                 self.config['batch_size']))

      sdqc_corr, sdqc_total, veracity_corr, veracity_total = 0, 0, 0, 0
      for _ in range(batch_num):
        X, X_pret, Y_sdqc, Y_veracity, sent_length, branch_length = \
            test_data.get_next(batch_num)
        c1, t1, c2, t2 = sess.run(
            [model.sdqc_correct_count, model.sdqc_total_count,
             model.veracity_correct_count, model.veracity_total_count],
            feed_dict={
                model.word_ids: X,
                model.word_ids_pret: X_pret,
                model.sdqc_labels: Y_sdqc,
                model.veracity_labels: Y_veracity,
                model.sent_length: sent_length,
                model.branch_length: branch_length,
            })
        sdqc_corr += c1
        sdqc_total += t1
        veracity_corr += c2
        veracity_total += t2

      utils.print_log('SDQC Task Acc = {}, Veracity Task Acc = {}'
                      .format(float(sdqc_corr) / sdqc_total,
                              float(veracity_corr) / veracity_total))
