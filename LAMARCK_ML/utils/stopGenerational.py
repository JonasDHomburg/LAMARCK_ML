from LAMARCK_ML.models import ModellUtil, NEADone


class StopByGenerationIndex(ModellUtil):
  arg_GENERATIONS = 'index'

  def __init__(self, **kwargs):
    self.max_t = kwargs.get(self.arg_GENERATIONS, 100)

  def end_select(self, func):
    def wrapper(model):
      print(model.abstract_timestamp)
      if model.abstract_timestamp >= self.max_t:
        raise NEADone()
      func()

    return wrapper
