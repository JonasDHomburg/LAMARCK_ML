from LAMARCK_ML.architectures.functions import Dense
from LAMARCK_ML.architectures.losses import Reduce, LossInterface
from LAMARCK_ML.architectures import OverParameterizedNeuralNetwork
from LAMARCK_ML.architectures.losses import SoftmaxCrossEntropyWithLogits, MeanSquaredError
from LAMARCK_ML.data_util import IOLabel, DimNames, TypeShape
from LAMARCK_ML.individuals.implementations.networkIndividualInterface import NetworkIndividualInterface
from LAMARCK_ML.reproduction.methods import Mutation, Recombination
from LAMARCK_ML.nn_framework import NeuralNetworkFrameworkInterface
from LAMARCK_ML.individuals.implementations.NetworkIndividual_pb2 import NetworkIndividualProto


class ClassifierIndividualOPACDG(NetworkIndividualInterface, Mutation.Interface, Recombination.Interface):
  arg_MAX_NN_DEPTH = 'max_depth'
  arg_MIN_NN_DEPTH = 'min_depth'
  arg_MAX_NN_BRANCH = 'max_branch'
  arg_NN_FUNCTIONS = 'functions'

  def __init__(self, **kwargs):
    super(ClassifierIndividualOPACDG, self).__init__(**kwargs)
    if len(self._networks) > 1:
      raise Exception('Expected 1 or 0 networks got: ' + str(len(self._networks)))
    elif len(self._networks) == 1:
      self.network = self._networks[0]
    else:
      _input = (IOLabel.DATA, *self._data_nts[IOLabel.DATA])
      _output = {IOLabel.TARGET: self._data_nts[IOLabel.TARGET][0]}
      _input = {'NN_DATA': _input}
      self.network = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: _input,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: _output,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: kwargs.get(self.arg_NN_FUNCTIONS, [Dense]),
        OverParameterizedNeuralNetwork.arg_MAX_DEPTH: kwargs.get(self.arg_MAX_NN_DEPTH, 7),
        OverParameterizedNeuralNetwork.arg_MIN_DEPTH: kwargs.get(self.arg_MIN_NN_DEPTH, 2),
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: kwargs.get(self.arg_MAX_NN_BRANCH, 1)
      })
      self._networks.append(self.network)

    if len(self._losses) != 0:
      raise Exception('Expected no loss!')
    _output = self._data_nts[IOLabel.TARGET][0]
    _output_units = _output.shape[DimNames.UNITS]
    if _output_units == 1:
      self.loss = MeanSquaredError(**{
        LossInterface.arg_REDUCE: Reduce.MEAN,
      })
    else:
      self.loss = SoftmaxCrossEntropyWithLogits(**{
        LossInterface.arg_REDUCE: Reduce.MEAN
      })
    self._losses.append(self.loss)

  def _cls_setstate(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _individual = NetworkIndividualProto()
      _individual.ParseFromString(state)
    elif isinstance(state, NetworkIndividualProto):
      _individual = state
    else:
      return

    self._networks = list()
    for network in _individual.networks:
      _obj = OverParameterizedNeuralNetwork.__new__(OverParameterizedNeuralNetwork)
      _obj.__setstate__(network)
      self._networks.append(_obj)
    self._data_nts = dict([(d.label, (TypeShape.from_pb(d.tsp), d.id_name)) for d in _individual.data_sources])
    self._losses = list()
    for loss in _individual.losses:
      _obj = LossInterface.__new__(LossInterface)
      _obj.__setstate__(loss)
      self._losses.append(_obj)

    super(NetworkIndividualInterface, self)._cls_setstate(_individual.baseIndividual)

    if len(self._networks) != 1:
      raise Exception('Restored individual has an invalid number of networks: ' + str(len(self._networks)))
    self.network = self._networks[0]
    if len(self._losses) != 1:
      raise Exception('Restored individual has an invalid number of losses: ' + str(len(self._losses)))
    self.loss = self._losses[0]

  def __eq__(self, other):
    if (super(ClassifierIndividualOPACDG, self).__eq__(other)
        and self.loss == other.loss
        and self.network == other.network
    ):
      return True
    return False

  def mutate(self, prob):
    result = ClassifierIndividualOPACDG.__new__(ClassifierIndividualOPACDG)
    result.metrics = dict()
    result.attr = dict(self.attr)
    result._data_nts = {label: (nts.__copy__(), id_name) for label, (nts, id_name) in self._data_nts.items()}
    result._losses = list(self._losses)
    result.loss = self.loss
    result.network = self.network.mutate(prob=prob)[0]
    result._networks = [result.network]
    result._id_name = self.getNewName()
    return [result]

  def recombine(self, other):
    result = ClassifierIndividualOPACDG.__new__(ClassifierIndividualOPACDG)
    result.metrics = dict()
    result.attr = dict(self.attr)
    result._data_nts = {label: (nts.__copy__(), id_name) for label, (nts, id_name) in self._data_nts.items()}
    result._losses = list(self._losses)
    result.loss = self.loss
    result.network = self.network.recombine(other.network)[0]
    result._networks = [result.network]
    result._id_name = self.getNewName()
    return [result]

  def norm(self, other):
    return self.network.norm(other.network)

  def update_state(self, *args, **kwargs):
    kwargs[OverParameterizedNeuralNetwork.meta_QUALITY] = self.metrics
    self.network.cmp = kwargs.pop(NeuralNetworkFrameworkInterface.arg_CMP)
    self.network.update_state(*args, **kwargs)