from itertools import product

from LAMARCK_ML.data_util import IOLabel
from LAMARCK_ML.reproduction.methods import Mutation, Recombination, RandomStep
from LAMARCK_ML.architectures.losses import Reduce, LossInterface
from LAMARCK_ML.architectures.losses import SoftmaxCrossEntropyWithLogits, MeanSquaredError
from LAMARCK_ML.individuals.implementations.networkIndividualInterface import NetworkIndividualInterface
from LAMARCK_ML.individuals.implementations.NetworkIndividual_pb2 import NetworkIndividualProto
from LAMARCK_ML.architectures.weightAgnosticNN import WeightAgnosticNeuralNetwork
from LAMARCK_ML.data_util.attribute import attr2pb, pb2attr
from LAMARCK_ML.data_util import TypeShape, Shape, DimNames
from LAMARCK_ML.architectures.functions import Perceptron
from LAMARCK_ML.metrics import Accuracy


class WeightAgnosticIndividual(NetworkIndividualInterface,
                               Recombination.Interface,
                               Mutation.Interface,
                               RandomStep.Interface,

                               Accuracy.Interface,
                               ):
  arg_WEIGHTS = 'test_weights'
  arg_NODES = 'nodes'
  arg_INITIAL_DEPTH = 'initial_depth'

  def __init__(self, **kwargs):
    super(WeightAgnosticIndividual, self).__init__(**kwargs)
    if len(self._networks) > 1:
      raise Exception('Expected 1 or 0 networks got: ' + str(len(self._networks)))
    elif len(self._networks) == 1:
      self.network = self._networks[0]
    else:
      _input = self._data_nts[IOLabel.DATA]
      _output = self._data_nts[IOLabel.TARGET]
      in_name = _input[1]
      shapes = list()
      batch = _input[0].shape[DimNames.BATCH]
      has_batch = False
      dtype = _input[0].dtype
      for dim in _input[0].shape.dim:
        if dim.name != DimNames.BATCH:
          shapes.append(list(range(dim.size)))
        else:
          has_batch = True
      _input = dict()
      for p in product(*shapes):
        key = ':'.join([str(i) for i in p])
        _input[key] = (IOLabel.DATA,
                       TypeShape(dtype, Shape((DimNames.UNITS, 1) if not has_batch else
                                              (DimNames.BATCH, batch), (DimNames.UNITS, 1))),
                       in_name + '_' + key)

      shapes = list()
      batch = _output[0].shape[DimNames.BATCH]
      has_batch = False
      dtype = _output[0].dtype
      for dim in _output[0].shape.dim:
        if dim.name != DimNames.BATCH:
          shapes.append(list(range(dim.size)))
        else:
          has_batch = True
      _output = dict()
      for p in product(*shapes):
        _output[':'.join([str(i) for i in p])] = \
          TypeShape(dtype, Shape((DimNames.UNITS, 1) if not has_batch else
                                 (DimNames.BATCH, batch), (DimNames.UNITS, 1)))

      self.network = WeightAgnosticNeuralNetwork(**{
        WeightAgnosticNeuralNetwork.arg_INPUTS: _input,
        WeightAgnosticNeuralNetwork.arg_OUTPUT_TARGETS: _output,
        WeightAgnosticNeuralNetwork.arg_FUNCTIONS: kwargs.get(self.arg_WEIGHTS, [Perceptron]),
        WeightAgnosticNeuralNetwork.arg_INITIAL_DEPTH: kwargs.get(self.arg_INITIAL_DEPTH, 1),
      })
      self._networks.append(self.network)

    weights = kwargs.get(self.arg_WEIGHTS)
    if weights is None or not (isinstance(weights, list) and all([isinstance(w, float) for w in weights])):
      weights = [i - 2 for i in range(5)]
    self.attr[self.arg_WEIGHTS] = weights

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

  def __sub__(self, other):
    if not isinstance(other, self.__class__):
      return -1
    return self.network - other.network

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
      _obj = WeightAgnosticNeuralNetwork.__new__(WeightAgnosticNeuralNetwork)
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
    if (super(WeightAgnosticIndividual, self).__eq__(other)
        and self.loss == other.loss
        and self.network == other.network
    ):
      return True
    return False

  def norm(self, other):
    if not isinstance(other, self.__class__):
      return 0
    return self.network.norm(other.network)

  def update_state(self, *args, **kwargs):
    self.network.update_state(*args, **kwargs)

  def mutate(self, prob):
    result = WeightAgnosticIndividual.__new__(WeightAgnosticIndividual)
    result.metrics = dict()
    result.attr = dict([pb2attr(attr2pb(key, value)) for key, value in self.attr.items()])
    result._data_nts = {label: (nts.__copy__(), id_name) for label, (nts, id_name) in self._data_nts.items()}
    result._losses = list(self._losses)
    result.loss = self.loss
    result._networks = self.network.mutate(prob=prob)
    result.network = result._networks[0]
    result._id_name = self.getNewName()
    return [result]

  def step(self, step_size):
    result = WeightAgnosticIndividual.__new__(WeightAgnosticIndividual)
    result.metrics = dict()
    result.attr = dict([pb2attr(attr2pb(key, value)) for key, value in self.attr.items()])
    result._data_nts = {label: (nts.__copy__(), id_name) for label, (nts, id_name) in self._data_nts.items()}
    result._losses = list(self._losses)
    result.loss = self.loss
    result._networks = self.network.step(step_size=step_size)
    result.network = result._networks[0]
    result._id_name = self.getNewName()
    return [result]

  def recombine(self, other):
    result = WeightAgnosticIndividual.__new__(WeightAgnosticIndividual)
    result.metrics = dict()
    result.attr = dict([pb2attr(attr2pb(key, value)) for key, value in self.attr.items()])
    result._data_nts = {label: (nts.__copy__(), id_name) for label, (nts, id_name) in self._data_nts.items()}
    result._losses = list(self._losses)
    result.loss = self.loss
    result._networks = self.network.recombine(other.network)
    result.network = result._networks[0]
    result._id_name = self.getNewName()
    return [result]

  def build_instance(self, nn_framework):
    nn_framework.init_model()
    for f in self.network.functions:
      nn_framework.add_function(f)
    nn_framework.set_train_parameters(**{
      nn_framework.arg_LOSS: self.loss.__class__,
    })
    nn_framework.finalize_model(output_ids=self.network.output_mapping.values())
    # nn_framework.train() # This individual doesn't need to be trained

  def train_instance(self, nn_framework):
    return dict()

  def accuracy(self, nn_framework):
    acc = 0
    weights = self.attr.get(self.arg_WEIGHTS, [])
    for w in weights:
      nn_framework.set_weights(**{
        f.id_name: w for f in self.network.functions
      })
      acc += nn_framework.accuracy(self)
    return acc / len(weights)
