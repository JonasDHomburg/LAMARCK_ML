from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.data_util.typeShape import TypeShape
from LAMARCK_ML.architectures import NeuralNetwork
from LAMARCK_ML.architectures.losses import LossInterface
from LAMARCK_ML.individuals.implementations.NetworkIndividual_pb2 import NetworkIndividualProto
from LAMARCK_ML.metrics.implementations import FlOps, Parameters

import warnings


class NetworkIndividualInterface(IndividualInterface,
                                 FlOps.Interface,
                                 Parameters.Interface):
  arg_NEURAL_NETWORKS = 'neuralNetworks'  # TODO: change to more general Architecture
  arg_DATA_NTS = 'data_nts'

  def __init__(self, **kwargs):
    super(NetworkIndividualInterface, self).__init__(**kwargs)
    self._networks = kwargs.get(self.arg_NEURAL_NETWORKS, [])
    for idx, network in enumerate(self._networks):
      if not isinstance(network, NeuralNetwork):
        raise Exception(
          'False network provided! Expected instance of NeuralNetwork, got: ' + str(
            type(network)) + ' at idx: ' + str(idx))
    self._data_nts = kwargs.get(self.arg_DATA_NTS, dict())

    if not isinstance(self._data_nts, dict):
      raise Exception('Expected dict for ' + self.arg_DATA_NTS + ' but got ' + str(type(self._data_nts)))
    for label, (nts, data_set_id) in self._data_nts.items():
      if not (isinstance(data_set_id, str)
              and isinstance(nts, TypeShape)
              and isinstance(label, str)
              and label):
        raise Exception('Expected list of (TypeShape,id_name) but got: (' + str(type(nts)) + ', ' + str(
          type(data_set_id)) + ')')
    self._losses = kwargs.get(self.arg_LOSSES, [])
    for idx, loss in enumerate(self._losses):
      if not isinstance(loss, LossInterface):
        raise Exception(
          'False loss provided! Expected instance of LossInterface, got: ' + str(type(loss)) + ' at idx: ' + str(idx))

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
      _obj = NeuralNetwork.__new__(NeuralNetwork)
      _obj.__setstate__(network)
      self._networks.append(_obj)
    self._data_nts = dict([(d.label, (TypeShape.from_pb(d.tsp), d.id_name)) for d in _individual.data_sources])
    self._losses = list()
    for loss in _individual.losses:
      _obj = LossInterface.__new__(LossInterface)
      _obj.__setstate__(loss)
      self._losses.append(_obj)

    super(NetworkIndividualInterface, self)._cls_setstate(_individual.baseIndividual)

  def get_pb(self, result=None):
    if not isinstance(result, NetworkIndividualProto):
      result = NetworkIndividualProto()
    super(NetworkIndividualInterface, self).get_pb(result.baseIndividual)

    result.cls_name = self.__class__.__name__

    result.networks.extend([network.get_pb() for network in self._networks])
    result.data_sources.extend(
      [NetworkIndividualProto.DataSourceProto(id_name=id_name, label=label, tsp=nts.get_pb())
       for label, (nts, id_name) in self._data_nts.items()])
    result.losses.extend([loss.get_pb() for loss in self._losses])
    return result

  def flops_per_sample(self):
    result = 0
    for n in self._networks:
      if isinstance(n, FlOps.Interface):
        result += n.flops_per_sample()
      else:
        warnings.warn('Skipping network {} since it does not implement flops interface!'.format(str(n)), Warning)
    return result

  def parameters(self):
    result = 0
    for n in self._networks:
      if isinstance(n, Parameters.Interface):
        result += n.parameters()
      else:
        warnings.warn('Skipping network {} since it does not implement parameter interface!'.format(str(n)), Warning)
    return result
