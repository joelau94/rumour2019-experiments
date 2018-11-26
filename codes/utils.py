import datetime


def print_log(s):
  print('[{}] {}'.format(str(datetime.datetime.now()), s))