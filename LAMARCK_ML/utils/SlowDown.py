from LAMARCK_ML.models import ModellUtil
import time

class SlowDown(ModellUtil):
  arg_SLEEP_TIME = 'sleep_time'
  def __init__(self, **kwargs):
    super(SlowDown, self).__init__(**kwargs)
    self.sleep_time = kwargs.get(self.arg_SLEEP_TIME, 0)

  def end_replace(self, func):
    def wrapper(model):
      func()
      if self.sleep_time > 0:
        time.sleep(self.sleep_time)
    return wrapper
