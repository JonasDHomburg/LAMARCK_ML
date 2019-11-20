from LAMARCK_ML.models import ModellUtil, NEADone
from LAMARCK_ML.utils import SortingClass


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

  class MetricContainer():
    pass

  def __init__(self, **kwargs):
    super(StopByNoProgress, self).__init__(**kwargs)
    self.patience = kwargs.get(self.arg_PATIENCE, 5)
    self.cmp = kwargs.get(self.arg_CMP)
    self.best_ind_sc = None
    self.best_ind_metrics = None
    self.waiting = 0

  def end_evaluate(self, func):
    def wrapper(model):
      # new_best_ind_sc = StopByNoProgress.MetricContainer()
      new_best_ind_sc = max([SortingClass(obj=ind, cmp=self.cmp) for ind in model.generation])
      new_best_metrics = dict(new_best_ind_sc.obj.metrics)
      if (self.best_ind_sc is None or (self.cmp is not None and self.cmp(new_best_metrics, self.best_ind_metrics)) or
         new_best_ind_sc > self.best_ind_sc):
        self.waiting = 0
        self.best_ind_sc = new_best_ind_sc
        self.best_ind_metrics = new_best_metrics
      else:
        self.waiting += 1
      if self.waiting > self.patience:
        raise NEADone()
      func()

    return wrapper
