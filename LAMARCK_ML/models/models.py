import types
from enum import IntEnum

from LAMARCK_ML.individuals import IndividualInterface
from LAMARCK_ML.metrics.interface import MetricInterface
from LAMARCK_ML.models.GenerationalModel_pb2 import GenerationalModelProto
from LAMARCK_ML.models.interface import ModelInterface, NEADone, \
  NoSelectionMethod, NoReplaceMethod, NoReproductionMethod, NoMetric, ModellUtil
from LAMARCK_ML.replacement.interface import ReplacementSchemeInterface
from LAMARCK_ML.reproduction import AncestryEntity
from LAMARCK_ML.reproduction.methods import MethodInterface
from LAMARCK_ML.selection.interface import SelectionStrategyInterface
from LAMARCK_ML.utils.evaluation import EvaluationHelperInterface


class GenerationalModel(ModelInterface):
  pb_GENERATION = b'generation'
  pb_SELECTION = b'selection'
  pb_REPRODUCTION_POOLS = b'reproduction_pools'
  pb_END_INDIVIDUAL = b'#end_individual'

  class State(IntEnum):
    UNINITIALIZED = 0,
    INITIALIZED = 1,
    EVALUATE = 2,
    SELECT = 3,
    REPRODUCE = 4,
    REPLACE = 5,
    DONE = 6,

  OVERRIDEABLE_FUNCTIONS = [
    'end_prepare',
    'end_evaluate',
    'end_select',
    'end_replace',
    'end_reproduce',
    'end_reproduction_step',
    'seed_generation',
    'nea_done'
  ]

  def __init__(self, **kwargs):
    self._GENERATION = None
    self._SELECTION = None
    self._REPRODUCTION = None
    self._REPRODUCTION_POOLS = None
    self._GENERATION_IDX = -1

    self._utils = list()
    self._metrics = list()
    self._selection_strategy = None
    self.reproduction_methods = list()
    self._replacement_scheme = None
    self._evaluation_helper = None

    self.data = kwargs.get('datasets')
    self.running = False

    for funcName in GenerationalModel.OVERRIDEABLE_FUNCTIONS:
      setattr(self, '_' + funcName, types.MethodType(getattr(self, '_' + funcName + '_'), self))
    self._STATE = GenerationalModel.State.UNINITIALIZED
    pass

  def _end_prepare_(self, model):
    pass

  def _end_evaluate_(self, model):
    pass

  def _end_select_(self, model):
    pass

  def _end_replace_(self, model):
    pass

  def _end_reproduce_(self, model):
    pass

  def _end_reproduction_step_(self, model):
    pass

  def _seed_generation_(self, model):
    pass

  def _nea_done_(self, model):
    pass

  def add(self, functionality, *args):
    def add_func(func):
      if isinstance(func, type) and issubclass(func, MetricInterface):
        self._metrics.append(func())
        return False

      if isinstance(func, MetricInterface):
        self._metrics.append(func)
        return False

      if isinstance(func, SelectionStrategyInterface):
        self._selection_strategy = func
        return False

      if isinstance(func, MethodInterface):
        self.reproduction_methods.append(func)
        return False

      if isinstance(func, ReplacementSchemeInterface):
        self._replacement_scheme = func
        return False

      if isinstance(func, EvaluationHelperInterface):
        self._evaluation_helper = func
        return False

      if isinstance(func, ModellUtil):
        for funcName in GenerationalModel.OVERRIDEABLE_FUNCTIONS:
          tempFunc = getattr(func, funcName, None)
          if callable(tempFunc):
            currentFunc = getattr(self, '_' + funcName)
            setattr(self, '_' + funcName, types.MethodType(tempFunc(currentFunc), self))
        self._utils.append(func)
        return False
      return True

    if isinstance(functionality, list):
      return not any([add_func(f) for f in functionality])
    else:
      return not add_func(functionality)

    pass

  def reset(self):
    try:
      self._GENERATION = None
      self._SELECTION = None
      self._REPRODUCTION = None
      self._REPRODUCTION_POOLS = None
      self._GENERATION_IDX = -1
      self._seed_generation()
      self._STATE = GenerationalModel.State.INITIALIZED
      return True
    except:
      return False

  def remove(self, functionality):
    if isinstance(functionality, type) and issubclass(functionality, MetricInterface):
      try:
        self._metrics.remove(functionality())
      except Exception:
        Warning("Tried to remove non existing metric!")
        return False
      return True

    if isinstance(functionality, MetricInterface):
      try:
        self._metrics.remove(functionality)
      except Exception:
        Warning("Tried to remove non existing metric!")
        return False
      return True

    if isinstance(functionality, MethodInterface):
      try:
        self.reproduction_methods.remove(functionality)
      except Exception:
        Warning("Tried to remove non existing reproduction method")
        return False
      return True

    if isinstance(functionality, ModellUtil):
      try:
        self._utils.remove(functionality)
      except Exception:
        Warning(str(functionality) + "hasn't been assigned in beforehand!")
        return False
      for funcName in GenerationalModel.OVERRIDEABLE_FUNCTIONS:
        setattr(self, '_' + funcName, types.MethodType(getattr(self, '_' + funcName + '_'), self))
      for util_module in self._utils:
        for funcName in GenerationalModel.OVERRIDEABLE_FUNCTIONS:
          tempFunc = getattr(util_module, funcName, None)
          if callable(tempFunc):
            setattr(self, '_' + funcName, types.MethodType(tempFunc(getattr(self, '_' + funcName)), self))
      return True
    return False

  def _evaluate(self):
    self._evaluation_helper.evaluate(self._GENERATION, self._metrics)
    self._STATE = GenerationalModel.State.EVALUATE
    self._end_evaluate()

  def _select(self):
    self._SELECTION = self._selection_strategy.select(self._GENERATION)
    self._STATE = GenerationalModel.State.SELECT
    self._end_select()

  def _reproduce(self):
    current_pool = self._SELECTION
    self._REPRODUCTION = list()
    self._REPRODUCTION_POOLS = list()
    for method in self.reproduction_methods:
      current_pool, derivation = method.reproduce(current_pool)
      self._REPRODUCTION.append((method.ID, derivation))
      self._REPRODUCTION_POOLS.append(current_pool)
      self._end_reproduction_step()
    self._STATE = GenerationalModel.State.REPRODUCE
    self._end_reproduce()

  def _replace(self):
    self._GENERATION = self._replacement_scheme.new_generation(self._GENERATION, self._REPRODUCTION_POOLS)
    self._STATE = GenerationalModel.State.REPLACE
    self._end_replace()

  def _prepare_with_defaults(self):
    if len(self._metrics) <= 0:
      raise NoMetric("Please add a metric function.")
    if not isinstance(self._selection_strategy, SelectionStrategyInterface):
      raise NoSelectionMethod("Please assign a valid selection strategy to the model.")
    if len(self.reproduction_methods) <= 0:
      raise NoReproductionMethod("Please assign one or more reproduction rules to the model.")
    if not isinstance(self._replacement_scheme, ReplacementSchemeInterface):
      raise NoReplaceMethod("Please assign a replacement scheme.")

    if self._evaluation_helper is None:
      try:
        from LAMARCK_ML.utils.evaluation import LocalEH
        # TODO: pass nn-framework
        self._evaluation_helper = LocalEH()
      except:
        pass
    self._end_prepare()
    pass

  @property
  def generation(self):
    return self._GENERATION

  @generation.setter
  def generation(self, generation):
    if self._GENERATION_IDX < 0:
      self._GENERATION_IDX = 0
      self._GENERATION = generation

  @property
  def selection(self):
    return self._SELECTION

  @property
  def reproduction(self):
    return self._REPRODUCTION

  @property
  def generation_idx(self):
    return self._GENERATION_IDX

  @property
  def utils(self):
    return self._utils

  def run(self):
    def increment():
      self._GENERATION_IDX += 1
      self._STATE = GenerationalModel.State.INITIALIZED

    def stop():
      raise NEADone()

    transition = {
      GenerationalModel.State.UNINITIALIZED: self._evaluate,
      GenerationalModel.State.INITIALIZED: self._evaluate,
      GenerationalModel.State.EVALUATE: self._select,
      GenerationalModel.State.SELECT: self._reproduce,
      GenerationalModel.State.REPRODUCE: self._replace,
      GenerationalModel.State.REPLACE: increment,
      GenerationalModel.State.DONE: stop,
    }
    try:
      if (self._STATE == GenerationalModel.State.INITIALIZED or
          self._STATE == GenerationalModel.State.UNINITIALIZED):
        self._prepare_with_defaults()
      self.running = True
      while self.running:
        transition[self._STATE]()
    except NEADone:
      pass
    self._STATE = GenerationalModel.State.DONE
    self._nea_done()
    pass

  def stop(self):
    self.running = False

  @property
  def abstract_timestamp(self):
    return self._GENERATION_IDX

  def get_pb(self, result=None):
    if result is None:
      result = GenerationalModelProto()
    if self._GENERATION:
      result.generation.extend([ind.get_pb() for ind in self._GENERATION])
    if self._SELECTION:
      result.selection.extend([ind.get_pb() for ind in self._SELECTION])
    if self._REPRODUCTION:
      for method_id, ancestry in self._REPRODUCTION:
        reprodPB = GenerationalModelProto.ReproductionProto()
        reprodPB.method = method_id
        reprodPB.ancestry.extend([anc.get_pb() for anc in ancestry])
        result.reproduction.append(reprodPB)
    if self._REPRODUCTION_POOLS:
      for pool in self._REPRODUCTION_POOLS:
        poolPB = GenerationalModelProto.ReproductionPoolProto()
        poolPB.individuals.extend([ind.get_pb() for ind in pool])
        result.reproduction_pools.append(poolPB)
    result.generation_idx = self._GENERATION_IDX
    result.state = self._STATE.value
    return result

  def setstate_from_pb(self, _model_pb):
    _model = GenerationalModelProto()
    _model.ParseFromString(_model_pb)
    if len(_model.generation) > 0:
      self._GENERATION = [IndividualInterface.get_instance(ind) for ind in _model.generation]
    if len(_model.selection) > 0:
      self._SELECTION = [IndividualInterface.get_instance(ind) for ind in _model.selection]
    if len(_model.reproduction) > 0:
      self._REPRODUCTION = [(reproPB.method, [AncestryEntity.from_pb(anc) for anc in reproPB.ancestry])
                            for reproPB in _model.reproduction]
    if len(_model.reproduction_pools) > 0:
      self._REPRODUCTION_POOLS = [[IndividualInterface.get_instance(ind) for ind in poolPB.individuals]
                                  for poolPB in _model.reproduction_pools]
    if _model.generation_idx:
      self._GENERATION_IDX = _model.generation_idx
    if _model.state:
      self._STATE = GenerationalModel.State(_model.state)

  def state_stream(self):
    if self._GENERATION:
      yield self.pb_GENERATION
      for ind in self._GENERATION:
        y = ind.__getstate__()
        yield y
        yield b'\n'+self.pb_END_INDIVIDUAL
    yield b'\n\n'
    if self._SELECTION:
      yield self.pb_SELECTION
      for ind in self._SELECTION:
        yield ind.__getstate__()
        yield b'\n'+self.pb_END_INDIVIDUAL
    yield b'\n\n'
    if self._REPRODUCTION_POOLS:
      for pool in self._REPRODUCTION_POOLS:
        yield self.pb_REPRODUCTION_POOLS
        for ind in pool:
          yield ind.__getstate__()
          yield b'\n'+self.pb_END_INDIVIDUAL
        yield b'\n\n'
      yield b'\n'
    rest = GenerationalModelProto()
    if self._REPRODUCTION:
      for method_id, ancestry in self._REPRODUCTION:
        reprodPB = GenerationalModelProto.ReproductionProto()
        reprodPB.method = method_id
        reprodPB.ancestry.extend([anc.get_pb() for anc in ancestry])
        rest.reproduction.append(reprodPB)
    rest.generation_idx = self._GENERATION_IDX
    rest.state = self._STATE.value
    rest = rest.SerializeToString()
    test = GenerationalModel.__new__(GenerationalModel)
    test.setstate_from_pb(rest)
    GenerationalModel.debug = rest
    yield rest

  def from_state_stream(self, stream):
    line = next(stream)
    if line == self.pb_GENERATION + b'\n':
      self._GENERATION = list()
      line = next(stream)
      while line != b'\n':
        pb = bytearray(b'\n')
        while line != self.pb_END_INDIVIDUAL+b'\n':
          pb.extend(line)
          line = next(stream)
        self._GENERATION.append(IndividualInterface.get_instance(bytes(pb[:-1])))
        line = next(stream)
      line = next(stream)
    if line == self.pb_SELECTION + b'\n':
      self._SELECTION = list()
      line = next(stream)
      while line != b'\n':
        pb = bytearray(b'\n')
        while line != self.pb_END_INDIVIDUAL+b'\n':
          pb.extend(line)
          line = next(stream)
        self._SELECTION.append(IndividualInterface.get_instance(bytes(pb[:-1])))
        line = next(stream)
      line = next(stream)
    if line == self.pb_REPRODUCTION_POOLS + b'\n':
      self._REPRODUCTION_POOLS = list()
    while line == self.pb_REPRODUCTION_POOLS + b'\n':
      line = next(stream)
      new_pool = list()
      while line != b'\n':
        pb = bytearray(b'\n')
        while line != self.pb_END_INDIVIDUAL+b'\n':
          pb.extend(line)
          line = next(stream)
        new_pool.append(IndividualInterface.get_instance(bytes(pb[:-1])))
        line = next(stream)
      self._REPRODUCTION_POOLS.append(new_pool)
      line = next(stream)
    line = next(stream)
    pb = bytearray(b'')
    while line != b'':
      pb.extend(line)
      line = next(stream, b'')
    self.setstate_from_pb(bytes(pb))
