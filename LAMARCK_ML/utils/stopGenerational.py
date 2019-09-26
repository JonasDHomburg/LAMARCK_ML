from LAMARCK_ML.models import ModellUtil, NEADone
from LAMARCK_ML.individuals import sortingClass


class StopByGenerationIndex(ModellUtil):
  arg_GENERATIONS = 'index'

  def __init__(self, **kwargs):
    super(StopByGenerationIndex, self).__init__(**kwargs)
    self.max_t = kwargs.get(self.arg_GENERATIONS, 100)

  def end_select(self, func):
    def wrapper(model):
      if model.abstract_timestamp >= self.max_t:
        raise NEADone()
      func()

    return wrapper


class StopByNoProgress(ModellUtil):
  arg_PATIENCE = 'patience'
  arg_CMP = 'cmp'

  def __init__(self, **kwargs):
    super(StopByNoProgress, self).__init__(**kwargs)
    self.patience = kwargs.get(self.arg_PATIENCE, 5)
    self.cmp = kwargs.get(self.arg_CMP)
    self.best_ind_sc = None
    self.waiting = 0

  def end_evaluate(self, func):
    def wrapper(model):
      new_best_ind_sc = max([sortingClass(obj=ind, cmp=self.cmp) for ind in model.generation])
      if (self.best_ind_sc is None or
          new_best_ind_sc > self.best_ind_sc):
        self.waiting = 0
        self.best_ind_sc = new_best_ind_sc
      else:
        self.waiting += 1
      if self.waiting > self.patience:
        raise NEADone()
      func()

    return wrapper
