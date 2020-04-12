from LAMARCK_ML.architectures.functions import Dense
from LAMARCK_ML.architectures.losses import Reduce, LossInterface
from LAMARCK_ML.architectures.losses import SoftmaxCrossEntropyWithLogits, MeanSquaredError
from LAMARCK_ML.architectures.neuralNetwork import NeuralNetwork
from LAMARCK_ML.architectures.functions import Softmax
from LAMARCK_ML.data_util import IOLabel, DimNames
from LAMARCK_ML.individuals.implementations.networkIndividualInterface import NetworkIndividualInterface
from LAMARCK_ML.reproduction.methods import Mutation, Recombination


class ClassifierIndividualACDG(NetworkIndividualInterface, Mutation.Interface, Recombination.Interface):
  arg_MAX_NN_DEPTH = 'max_depth'
  arg_MIN_NN_DEPTH = 'min_depth'
  arg_MAX_NN_BRANCH = 'max_branch'
  arg_NN_FUNCTIONS = 'functions'

  def __init__(self, **kwargs):
    super(ClassifierIndividualACDG, self).__init__(**kwargs)
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
    super(ClassifierIndividualACDG, self)._cls_setstate(_individual)

    if len(self._networks) != 1:
      raise Exception('Restored individual has an invalid number of networks: ' + str(len(self._networks)))
    self.network = self._networks[0]
    if len(self._losses) != 1:
      raise Exception('Restored individual has an invalid number of losses: ' + str(len(self._losses)))
    self.loss = self._losses[0]

  def __eq__(self, other):
    if (super(ClassifierIndividualACDG, self).__eq__(other)
        and self.loss == other.loss
        and self.network == other.network
    ):
      return True
    return False

  def mutate(self, prob):
    result = ClassifierIndividualACDG.__new__(ClassifierIndividualACDG)
    pb = self.get_pb()
    result.__setstate__(pb)
    result.network = self.network.mutate(prob=prob)[0]
    result._networks = [result.network]
    result._id_name = self.getNewName()
    return [result]

  def recombine(self, other):
    result = ClassifierIndividualACDG.__new__(ClassifierIndividualACDG)
    pb = self.get_pb()
    result.__setstate__(pb)
    result.network = self.network.recombine(other.network)[0]
    result._networks = [result.network]
    result._id_name = self.getNewName()
    return [result]

  def norm(self, other):
    return self.network.norm(other.network)

  def update_state(self, *args, **kwargs):
    self.network.update_state(*args, **kwargs)

  def build_instance(self, nn_framework):
    nn_framework.init_model({IOLabel.DATA}, {IOLabel.TARGET})
    f_id2obj = dict()
    for f in self.network.functions:
      nn_framework.add_function(f)
      f_id2obj[f.id_name] = f
    nn_framework.set_train_parameters(**{
      nn_framework.arg_LOSS: self.loss.__class__,
    })
    softmax_out = list()
    for label, f_id in self.network.output_mapping.values():
      f_obj = f_id2obj[f_id]
      softmax = Softmax(**Softmax.generateParameters(
        input_dict={IOLabel.SOFTMAX_IN: (label, f_obj.outputs, f_id)},
        expected_outputs={IOLabel.SOFTMAX_OUT: f_obj.outputs[label]},
      )[0][0])
      nn_framework.add_function(softmax)
      softmax_out.append((IOLabel.SOFTMAX_OUT, softmax.id_name))
    nn_framework.finalize_model(output_ids=softmax_out)

  def train_instance(self, nn_framework):
    return nn_framework.train()
