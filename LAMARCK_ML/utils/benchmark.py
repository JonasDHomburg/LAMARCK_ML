from LAMARCK_ML.models.interface import ModellUtil
from datetime import datetime

class Benchmark(ModellUtil):
  def __init__(self, **kwargs):
    super(Benchmark, self).__init__(**kwargs)
    self.last_time_stamp = datetime.now()

  def print(self, func, func_name):
    print('%s: %s' % (func_name, datetime.now() - self.last_time_stamp))
    before = datetime.now()
    func()
    end = datetime.now()
    print('Hooks: %s' % (end - before))
    print('============')
    self.last_time_stamp = datetime.now()

  def end_prepare(self, func):
    def wrapper(model):
      print('%s: %s' % ('Prepare', datetime.now() - self.last_time_stamp))
      before = datetime.now()
      func()
      end = datetime.now()
      print('Hooks: %s' % (end - before))
      print('============')
      self.last_time_stamp = datetime.now()
    return wrapper

  def end_evaluate(self, func):
    def wrapper(model):
      self.print(func, 'Evaluate')
    return wrapper

  def end_select(self, func):
    def wrapper(model):
      self.print(func, 'Select')
    return wrapper

  def end_replace(self, func):
    def wrapper(model):
      self.print(func, 'Replace')
    return wrapper

  def end_reproduce(self, func):
    def wrapper(model):
      self.print(func, 'Reproduce')
    return wrapper

  def end_reproduce_step(self, func):
    def wrapper(model):
      self.print(func, 'Reproduce Step')
    return wrapper

  def new_done(self, func):
    def wrapper(model):
      self.print(func, 'Done')
    return wrapper