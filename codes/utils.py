import datetime
import sys


def print_log(s):
  print('[{}] {}'.format(str(datetime.datetime.now()), s))
  sys.stdout.flush()