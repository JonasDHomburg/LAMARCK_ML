from LAMARCK_ML.architectures.functions import Dense
from LAMARCK_ML.architectures.losses import Reduce, LossInterface
from LAMARCK_ML.architectures.losses import SoftmaxCrossEntropyWithLogits, MeanSquaredError
from LAMARCK_ML.architectures.neuralNetwork import NeuralNetwork
from LAMARCK_ML.data_util import IOLabel, DimNames
from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.reproduction.methods import Mutation, Recombination


class ClassifierIndividual(IndividualInterface, Mutation.Interface, Recombination.Interface):
  arg_MAX_NN_DEPTH = 'max_depth'
  arg_MIN_NN_DEPTH = 'min_depth'
  arg_MAX_NN_BRANCH = 'max_branch'
  arg_NN_FUNCTIONS = 'functions'

  def __init__(self, **kwargs):
    super(ClassifierIndividual, self).__init__(**kwargs)
    if len(self._networks) > 1:
      raise Exception('Expected 1 or 0 networks got: ' + str(len(self._networks)))
    elif len(self._networks) == 1:
      self.network = self._networks[0]
    else:
      _input = (IOLabel.DATA, *self._data_nts[IOLabel.DATA])
      _output = {IOLabel.TARGET: self._data_nts[IOLabel.TARGET][0]}
      _input = {'NN_DATA': _input}
      self.network = NeuralNetwork(**{
        NeuralNetwork.arg_INPUTS: _input,
        NeuralNetwork.arg_OUTPUT_TARGETS: _output,
        NeuralNetwork.arg_FUNCTIONS: kwargs.get(self.arg_NN_FUNCTIONS, [Dense]),
        NeuralNetwork.arg_MAX_DEPTH: kwargs.get(self.arg_MAX_NN_DEPTH, 7),
        NeuralNetwork.arg_MIN_DEPTH: kwargs.get(self.arg_MIN_NN_DEPTH, 2),
        NeuralNetwork.arg_MAX_BRANCH: kwargs.get(self.arg_MAX_NN_BRANCH, 1)
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

  def _cls_setstate(self, _individual):
    super(ClassifierIndividual, self)._cls_setstate(_individual)

    if len(self._networks) != 1:
      raise Exception('Restored individual has an invalid number of networks: ' + str(len(self._networks)))
    self.network = self._networks[0]
    if len(self._losses) != 1:
      raise Exception('Restored individual has an invalid number of losses: ' + str(len(self._losses)))
    self.loss = self._losses[0]

  def __eq__(self, other):
    if (isinstance(other, self.__class__)
        and self._id_name == other._id_name
        and self.loss == other.loss
        and self.network == other.network
        and self.metrics == other.metrics
        and len(self.attr) == len(other.attr) == len(
          [value == other.attr(key) for key, value in self.attr.items()])
    ):
      return True
    return False

  def mutate(self, prob):
    result = ClassifierIndividual.__new__(ClassifierIndividual)
    pb = self.get_pb()
    new_networks = [network.mutate(prob)[0] for network in self._networks]
    pb.ClearField("networks")
    pb.networks.extend([network.get_pb() for network in new_networks])
    result.__setstate__(pb)
    return [result]

  def recombine(self, other):
    pass

  def norm(self, other):
    return self.network.norm(other.network)
