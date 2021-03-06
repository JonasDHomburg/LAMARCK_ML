import os
import types

from LAMARCK_ML.models.interface import ModellUtil


class ModelStateSaverLoader(ModellUtil):
  arg_FILE = 'file'
  arg_PREPARATION = 'preparation'
  arg_EVALUATION = 'evaluation'
  arg_SELECTION = 'selection'
  arg_REPRODUCTION = 'reproduction'
  arg_REPLACEMENT = 'replacement'
  arg_NEA_DONE = 'nea_done'
  arg_FILE_BATCH = 'file_batch'

  def __init__(self, **kwargs):
    super(ModelStateSaverLoader, self).__init__(**kwargs)
    self.file = kwargs.get(self.arg_FILE, './model_state.pb')
    self.file_batch = kwargs.get(self.arg_FILE_BATCH, 1)
    if kwargs.get(self.arg_PREPARATION, False):
      setattr(self, 'end_prepare', types.MethodType(ModelStateSaverLoader._end_prepare, self))
    if kwargs.get(self.arg_EVALUATION, True):
      setattr(self, 'end_evaluate', types.MethodType(ModelStateSaverLoader._end_evaluate, self))
    if kwargs.get(self.arg_SELECTION, False):
      setattr(self, 'end_select', types.MethodType(ModelStateSaverLoader._end_select, self))
    if kwargs.get(self.arg_REPLACEMENT, False):
      setattr(self, 'end_replace', types.MethodType(ModelStateSaverLoader._end_replace, self))
    if kwargs.get(self.arg_REPRODUCTION, False):
      setattr(self, 'end_reproduce', types.MethodType(ModelStateSaverLoader._end_reproduce, self))
    if kwargs.get(self.arg_NEA_DONE, False):
      setattr(self, 'nea_done', types.MethodType(ModelStateSaverLoader._nea_done, self))
    pass

  def save_model(self, model):
    with open(self.file + '_tmp', 'wb') as f:
      for data in model.state_stream():
        f.write(data)
    os.rename(self.file + '_tmp', self.file)

  def load_model(self, model):
    if os.path.isfile(self.file):
      with open(self.file, 'rb') as f:
        def stream(f):
          while True:
            lines = f.readlines(self.file_batch)
            if not lines:
              break
            for line in lines:
              yield line

        model.from_state_stream(stream(f))

  def _end_prepare(self, func):
    def wrapper(model):
      self.load_model(model)
      func()

    return wrapper

  def _end_evaluate(self, func):
    def wrapper(model):
      self.save_model(model)
      func()

    return wrapper

  def _end_select(self, func):
    def wrapper(model):
      self.save_model(model)
      func()

    return wrapper

  def _end_replace(self, func):
    def wrapper(model):
      self.save_model(model)
      func()

    return wrapper

  def _end_reproduce(self, func):
    def wrapper(model):
      self.save_model(model)
      func()

    return wrapper

  def _nea_done(self, func):
    def wrapper(model):
      self.save_model(model)
      func()

    return wrapper

  pass
