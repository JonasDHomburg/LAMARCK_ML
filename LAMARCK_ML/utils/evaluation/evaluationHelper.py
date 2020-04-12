from LAMARCK_ML.nn_framework import NeuralNetworkFrameworkInterface
from LAMARCK_ML.utils.evaluation.interface import EvaluationHelperInterface
from joblib import Parallel, delayed
import threading
import queue
import time


class BaseEH(EvaluationHelperInterface):
  def __init__(self, **kwargs):
    super(BaseEH, self).__init__()

  def evaluate(self, generation, metrics):
    for individual in generation:
      individual.metrics = dict([(m.ID, m.evaluate(individual=individual, framework=None)) for m in metrics])


class LocalEH(EvaluationHelperInterface):
  arg_NN_FRAMEWORK = 'framework'

  def __init__(self, **kwargs):
    super(LocalEH, self).__init__()
    self._framework = kwargs.get(self.arg_NN_FRAMEWORK)
    #   TODO: setup framework
    if not isinstance(self._framework, NeuralNetworkFrameworkInterface):
      raise Exception()

  def evaluate(self, generation, metrics):
    for individual in generation:
      individual.build_instance(self._framework)
      state = individual.train_instance(self._framework)
      individual.metrics = {m.ID: m.evaluate(individual=individual, framework=self._framework) for m in metrics}
      individual.update_state(**state)
      self._framework.reset()


class LocalParallelEH_joblib(EvaluationHelperInterface):
  arg_NN_FRAMEWORK_CLASS = 'framework_cls'
  arg_NN_FRAMEWORK_KWARGS = 'framework_kwargs'
  arg_PARALLEL = 'parallel'
  arg_ADAPT_KWARGS = 'adapt_kwargs'

  def __init__(self, **kwargs):
    super(LocalParallelEH_joblib, self).__init__()
    self._parallel = kwargs.get(self.arg_PARALLEL, 1)
    self._framework_cls = kwargs.get(self.arg_NN_FRAMEWORK_CLASS)
    self._framework_kwargs = kwargs.get(self.arg_NN_FRAMEWORK_KWARGS, dict())
    adapt_kwargs = kwargs.get(self.arg_ADAPT_KWARGS, [])
    self._framworks = list()
    for i in range(self._parallel):
      kwargs = {k: (v if k not in adapt_kwargs else v.format(i)) for k, v in self._framework_kwargs.items()}
      self._framworks.append(self._framework_cls(**kwargs))

  def evaluate(self, generation, metrics):
    def eval(ind):
      framework = self._framworks.pop(0)
      ind.build_instance(framework)
      state = ind.train_instance(framework)
      ind.metrics = {m.ID: m.evaluate(individual=ind, framework=framework) for m in metrics}
      ind.update_state(**state)
      framework.reset()
      self._framworks.append(framework)
      pass

    for _ in Parallel(n_jobs=self._parallel, require='sharedmem')(
        delayed(eval)(ind) for ind in generation
    ):
      pass


class LocalParallelEH_threading(EvaluationHelperInterface):
  arg_NN_FRAMEWORK_CLASS = 'framework_cls'
  arg_NN_FRAMEWORK_KWARGS = 'framework_kwargs'
  arg_PARALLEL = 'parallel'
  arg_ADAPT_KWARGS = 'adapt_kwargs'

  class nn_thread(threading.Thread):
    def __init__(self, q_in, q_out, framework, *args, **kwargs):
      super(LocalParallelEH_threading.nn_thread, self).__init__(*args, **kwargs)
      self.q_in = q_in
      self.q_out = q_out
      self.framework = framework

    def run(self):
      while True:
        try:
          ind, metrics = self.q_in.get()
          ind.build_instance(self.framework)
          ind_state = ind.train_instance(self.framework)
          ind_metrics = {m.ID: m.evaluate(individual=ind, framework=self.framework) for m in metrics}
          self.q_out.put((ind.id_name, ind_state, ind_metrics))
          self.framework.reset()
        except queue.Empty:
          continue

  def __init__(self, **kwargs):
    super(LocalParallelEH_threading, self).__init__()
    self._parallel = kwargs.get(self.arg_PARALLEL, 1)
    self._framework_cls = kwargs.get(self.arg_NN_FRAMEWORK_CLASS)
    self._framework_kwargs = kwargs.get(self.arg_NN_FRAMEWORK_KWARGS, dict())
    adapt_kwargs = kwargs.get(self.arg_ADAPT_KWARGS, [])
    self.q_eval = queue.Queue()
    self.q_res = queue.Queue()
    self._framworks = list()
    for i in range(self._parallel):
      kwargs = {k: (v if k not in adapt_kwargs else v.format(i)) for k, v in self._framework_kwargs.items()}
      f = LocalParallelEH_threading.nn_thread(self.q_eval, self.q_res, self._framework_cls(**kwargs))
      f.start()
      self._framworks.append(f)

  def evaluate(self, generation, metrics):
    waiting = dict()
    for ind in generation:
      self.q_eval.put((ind, metrics))
      waiting[ind.id_name] = ind

    while waiting:
      ind_id, ind_state, ind_metrics = self.q_res.get()
      ind = waiting.pop(ind_id)
      ind.metrics = ind_metrics
      ind.update_state(**ind_state)


class GraphLayoutEH(EvaluationHelperInterface):

  def __init__(self, **kwargs):
    super(GraphLayoutEH, self).__init__(**kwargs)

  def evaluate(self, generation, metrics):
    for individual in generation:
      individual.metrics = dict([(m.ID, m.evaluate(individual=individual, framework=None))
                                 for m in metrics])
