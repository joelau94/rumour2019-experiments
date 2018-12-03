from collections import defaultdict
import json
import sys


def read_text(prediction_file, id_file):
  ftext = open(prediction_file, 'r')
  fid = open(id_file, 'r')
  sdqc_ref = {}
  sdqc_hyp = {}
  veracity_ref = {}
  veracity_hyp = {}

  for line in ftext:
    if line.strip() == '':
      fid.readline()
      continue
    else:
      line = line.strip().split('|||')
      twid = fid.readline().strip()
      if len(line) == 5:
        veracity_ref[twid] = line[3].strip()
        veracity_hyp[twid] = line[4].strip()
        sdqc_ref[twid] = line[1].strip()
        sdqc_hyp[twid] = line[2].strip()
      elif len(line) == 3:
        sdqc_ref[twid] = line[1].strip()
        sdqc_hyp[twid] = line[2].strip()

  return sdqc_ref, sdqc_hyp, veracity_ref, veracity_hyp


def write_answer(sdqc_hyp, veracity_hyp, answer_file):
  for k, v in veracity_hyp.iteritems():
    if v == 'unverified':
      veracity_hyp[k] = ['false', 0.0]
    else:
      veracity_hyp[k] = [v, 1.0]
  ans = {
      'subtaskaenglish': sdqc_hyp,
      'subtaskbenglish': veracity_hyp
  }
  json.dump(ans, open(answer_file, 'w'))


def sdqc_confusion(ref, hyp):
  matrix = defaultdict(lambda: defaultdict(int))
  for k in ref.keys():
    matrix[ref[k]][hyp[k]] += 1
  sys.stdout.write('ref | hyp\tsupport\tdeny\tquery\tcomment\n')
  for r in ['support', 'deny', 'query', 'comment']:
    sys.stdout.write('{}\t{}\t{}\t{}\t{}\n'
                     .format(r,
                             matrix[r]['support'],
                             matrix[r]['deny'],
                             matrix[r]['query'],
                             matrix[r]['comment']))


def veracity_confusion(ref, hyp):
  matrix = defaultdict(lambda: defaultdict(int))
  for k in ref.keys():
    matrix[ref[k]][hyp[k]] += 1
  sys.stdout.write('ref | hyp\ttrue\tfalse\tunverified\n')
  for r in ['true', 'false', 'unverified']:
    sys.stdout.write('{}\t{}\t{}\t{}\n'
                     .format(r,
                             matrix[r]['true'],
                             matrix[r]['false'],
                             matrix[r]['unverified']))
