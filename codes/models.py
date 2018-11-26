"""Core Model"""
import cPickle as pkl

import numpy as np
import tensorflow as tf


class Embeddings(object):
  """Embeddings"""

  def __init__(self, embed_dim, vocab_size, reuse=True):

    self.embed_dim = embed_dim
    self.vocab_size = vocab_size
    self.reuse = reuse

    initializer = tf.zeros_initializer()
    with tf.variable_scope('Embeddings', reuse=self.reuse):
      self.emb = tf.get_variable(
          'emb',
          shape=(self.vocab_size, self.embed_dim),
          initializer=initializer
      )

  def load_pretrained_emb(self, embed_pret_file, dicts_file):
    dicts = pkl.load(open(dicts_file, 'r'))
    self.emb_nparray = np.zeros([len(dicts['w2i_pret']) + 1, self.embed_dim],
                                dtype=np.float32)

    for line in open(embed_pret_file, 'r'):
      we = line.strip().split()
      if we[0] in dicts['w2i_pret']:
        self.emb_nparray[dicts['w2i_pret'][we[0]]] = np.array(we[1:])

    with tf.variable_scope('Embeddings', reuse=self.reuse):
      self.emb_pret = tf.Variable(
          [0.0], trainable=False, name='emb_pret')
      self.emb_pret_placeholder = tf.placeholder(
          tf.float32, shape=self.emb_nparray.shape)
      self.embed_assign = tf.assign(
          self.emb_pret,
          self.emb_pret_placeholder,
          validate_shape=False)

  def init_pretrained_emb(self, sess):
    sess.run(self.embed_assign,
             feed_dict={self.emb_pret_placeholder: self.emb_nparray})

  def __call__(self, word_ids, word_ids_pret=None):
    embeddings = tf.nn.embedding_lookup(self.emb, word_ids)
    if word_ids_pret is not None:
      embeddings += tf.nn.embedding_lookup(self.emb_pret, word_ids_pret)
    return embeddings


class SelfAttn(object):
  """SelfAttn"""

  def __init__(self, attn_dim, reuse=True):
    self.attn_dim = attn_dim
    self.reuse = reuse

  def __call__(self, hidden_tape, masks):
    """
    hidden_tape: (batch, len, dim)
    """

    with tf.variable_scope('SelfAttn', reuse=self.reuse):
      q = tf.layers.dense(hidden_tape, units=self.attn_dim, name='q')
      k = tf.layers.dense(hidden_tape, units=self.attn_dim, name='k')
      v = hidden_tape

      normalizer = tf.rsqrt(tf.to_float(tf.shape(q)[-1]))
      # (batch, len, dim) x (batch, dim, len) --> (batch, len, len)
      logits = tf.matmul(q * normalizer, k, transpose_b=True)

      bias_mask = tf.expand_dims(-1e9 * (1. - masks), axis=-1)
      # (batch, len, 1) + (batch, len, len) --> (batch, len, len)
      logits += bias_mask

      attn_weights = tf.nn.softmax(logits)
      # (batch, len, len) x (batch, len, dim) --> (batch, len, dim)
      return tf.matmul(attn_weights, v)


class SentEncoder(object):
  """SentEncoder"""

  def __init__(self, hidden_dims, keep_prob=1.0, reuse=True):
    self.hidden_dims = hidden_dims
    self.keep_prob = keep_prob
    self.reuse = reuse

    with tf.variable_scope('SentEncoder', reuse=self.reuse):
      self.fw_step = self._step(hidden_dims)
      self.bw_step = self._step(hidden_dims)

  def _step(self, hidden_dims):
    with tf.variable_scope('SentEncoder', reuse=self.reuse):
      cells = [
          tf.contrib.rnn.LayerNormBasicLSTMCell(
              num_units=n, layer_norm=True, dropout_keep_prob=self.keep_prob)
          for n in hidden_dims
      ]
      return tf.contrib.rnn.MultiRNNCell(cells)

  def __call__(self, word_embeddings, sent_length):
    """
    word_embeddings: (batch, thread_max_len, sent_max_len, embed_dim)
    """
    # batch_size = tf.shape(word_embeddings)[0]
    # thread_max_len = tf.shape(word_embeddings)[1]
    # sent_max_len = tf.shape(word_embeddings)[2]
    # embed_dim = tf.shape(word_embeddings)[3]

    # word_embeddings = tf.reshape(
    #     word_embeddings,
    #     [batch_size * thread_max_len, sent_max_len, embed_dim])
    # sent_length = tf.reshape(sent_length, [-1])

    with tf.variable_scope('SentEncoder', reuse=self.reuse):

      outputs, states = tf.nn.bidirectional_dynamic_rnn(
          self.fw_step,
          self.bw_step,
          word_embeddings,
          sequence_length=sent_length,
          dtype=tf.float32,
          swap_memory=True
      )

      final_states = tf.concat([states[0][-1].h, states[1][-1].h], axis=-1)
      # final_states = tf.reshape(final_states,
      #                           [batch_size, thread_max_len, -1])

      return final_states


class BranchEncoder(object):
  """BranchEncoder"""

  def __init__(self, hidden_dims, attn_dim, keep_prob=1.0, reuse=True):
    self.hidden_dims = hidden_dims
    self.keep_prob = keep_prob
    self.reuse = reuse

    self.context_fn = SelfAttn(attn_dim, reuse=self.reuse)

    with tf.variable_scope('BranchEncoder', reuse=self.reuse):
      self.fw_step = self._step(hidden_dims)
      self.bw_step = self._step(hidden_dims)

  def _step(self, hidden_dims):
    with tf.variable_scope('BranchEncoder', reuse=self.reuse):
      cells = [
          tf.contrib.rnn.LayerNormBasicLSTMCell(
              num_units=n, layer_norm=True, dropout_keep_prob=self.keep_prob)
          for n in hidden_dims
      ]
      return tf.contrib.rnn.MultiRNNCell(cells)

  def __call__(self, sent_vecs, branch_length):
    with tf.variable_scope('BranchEncoder', reuse=self.reuse):

      outputs, states = tf.nn.bidirectional_dynamic_rnn(
          self.fw_step,
          self.bw_step,
          sent_vecs,
          sequence_length=branch_length,
          dtype=tf.float32,
          swap_memory=True
      )

      hidden_states = tf.concat(outputs, axis=-1)
      attn_context = self.context_fn(
          hidden_states,
          tf.sequence_mask(branch_length, dtype=tf.float32))

      return tf.concat([hidden_states, attn_context], axis=-1)


class SdqcClassifier(object):
  """SdqcClassifier"""

  def __init__(self, keep_prob=1.0, reuse=True):
    self.keep_prob = keep_prob
    self.reuse = reuse

  def __call__(self, sent_vecs, labels, masks):
    with tf.variable_scope('Sdqc', reuse=self.reuse):
      # (batch, len, 4)
      feats = tf.nn.dropout(sent_vecs, keep_prob=self.keep_prob)
      scores = tf.layers.dense(feats, units=4, name='scores')

    self.probabilities = tf.nn.softmax(scores, axis=-1)
    self.predictions = tf.argmax(self.probabilities, axis=-1)

    self.loss = tf.contrib.seq2seq.sequence_loss(scores, labels, masks)

    correct_count = tf.reduce_sum(
        tf.cast(tf.math.equal(self.predictions, labels),
                dtype=tf.float32) * masks)
    total_count = tf.reduce_sum(masks)

    return self.loss, self.predictions, self.probabilities, \
        correct_count, total_count


class VeracityClassifier(object):
  """VeracityClassifier"""

  def __init__(self, keep_prob=1.0, reuse=True):
    self.keep_prob = keep_prob
    self.reuse = reuse

  def __call__(self, sent_vecs, labels):
    with tf.variable_scope('Veracity', reuse=self.reuse):
      # (batch, 3)
      feats = tf.nn.dropout(sent_vecs, keep_prob=self.keep_prob)
      scores = tf.layers.dense(feats, units=3, name='scores')

    self.probabilities = tf.nn.softmax(scores, axis=-1)
    self.predictions = tf.argmax(self.probabilities, axis=-1)

    self.loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(labels, [-1]), logits=scores)

    correct_count = tf.reduce_sum(
        tf.cast(tf.math.equal(self.predictions, labels),
                dtype=tf.float32))
    total_count = tf.shape(labels)[0]

    return self.loss, self.predictions, self.probabilities, \
        correct_count, total_count


class RumourDetectModel(object):
  """RumourDetectModel"""

  def __init__(self,
               embed_dim,
               vocab_size,
               sent_hidden_dims,
               branch_hidden_dims,
               attn_dim,
               embed_pret_file=None,
               dicts_file=None,
               keep_prob=1.0,
               reuse=True):
    self.reuse = reuse
    self.embed_dim = embed_dim
    self.sent_vec_dim = sent_hidden_dims[-1] * 2

    self.embedder = Embeddings(embed_dim, vocab_size, reuse=self.reuse)
    if embed_pret_file:
      self.embedder.load_pretrained_emb(embed_pret_file, dicts_file)

    self.sent_encoder = SentEncoder(sent_hidden_dims,
                                    keep_prob=keep_prob,
                                    reuse=self.reuse)
    self.branch_encoder = BranchEncoder(branch_hidden_dims,
                                        attn_dim,
                                        keep_prob=keep_prob,
                                        reuse=self.reuse)
    self.sdqc_classifier = SdqcClassifier(keep_prob=keep_prob,
                                          reuse=self.reuse)
    self.veracity_classifier = VeracityClassifier(keep_prob=keep_prob,
                                                  reuse=self.reuse)

  def __call__(self, is_train=True):

    self.sent_length = tf.placeholder(dtype=tf.int64, shape=(None, None))
    self.branch_length = tf.placeholder(dtype=tf.int64, shape=(None,))
    self.word_ids = tf.placeholder(dtype=tf.int64, shape=(None, None, None))
    self.word_ids_pret = tf.placeholder(dtype=tf.int64,
                                        shape=(None, None, None))
    self.sdqc_labels = tf.placeholder(dtype=tf.int64, shape=(None, None))
    self.veracity_labels = tf.placeholder(dtype=tf.int64, shape=(None,))

    batch_size = tf.shape(self.word_ids)[0]
    thread_max_len = tf.shape(self.word_ids)[1]
    sent_max_len = tf.shape(self.word_ids)[2]

    word_ids = tf.reshape(self.word_ids,
                          [batch_size * thread_max_len, sent_max_len])
    word_ids_pret = tf.reshape(self.word_ids_pret,
                               [batch_size * thread_max_len, sent_max_len])

    word_embeddings = self.embedder(word_ids, word_ids_pret)

    sent_length = tf.reshape(self.sent_length, [-1])
    sent_vecs = self.sent_encoder(word_embeddings, sent_length)
    sent_vecs = tf.reshape(sent_vecs,
                           [batch_size, thread_max_len, self.sent_vec_dim])

    sent_vecs = self.branch_encoder(sent_vecs, self.branch_length)

    # (batch, len)
    branch_masks = tf.sequence_mask(self.branch_length, dtype=tf.float32)

    self.sdqc_loss, \
        self.sdqc_predictions, \
        self.sdqc_probabilities, \
        self.sdqc_correct_count, \
        self.sdqc_total_count = \
        self.sdqc_classifier(sent_vecs,
                             self.sdqc_labels,
                             branch_masks)
    self.veracity_loss, \
        self.veracity_predictions, \
        self.veracity_probabilities, \
        self.veracity_correct_count, \
        self.veracity_total_count = \
        self.veracity_classifier(sent_vecs[:, 0, :],
                                 self.veracity_labels)
