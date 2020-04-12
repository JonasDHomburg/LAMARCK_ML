from typing import Dict, List, Tuple, Set

from LAMARCK_ML.architectures.functions import Function
from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.reproduction.methods import Mutation, RandomStep, Recombination
from LAMARCK_ML.metrics import LayoutDistanceX, LayoutDistanceY, LayoutCrossingEdges
from LAMARCK_ML.nn_framework.interface import NeuralNetworkFrameworkInterface

from random import random, sample, seed,randint

seed()


class AncestryLayoutIndividual(IndividualInterface,
                               Mutation.Interface,
                               RandomStep.Interface,
                               Recombination.Interface,

                               LayoutCrossingEdges.Interface,
                               LayoutDistanceX.Interface,
                               LayoutDistanceY.Interface):

  def __init__(self, **kwargs):
    super(AncestryLayoutIndividual, self).__init__(**kwargs)
    self.node2X = dict()
    self.Y2node = dict()
    self.framework = None

  def __sub__(self, other):
    distance = 0
    for k in set(self.node2X.keys()).intersection(other.node2X.keys()):
      distance += (self.node2X[k] - other.node2X[k]) ** 2

  def norm(self, other):
    return self - other

  def update_state(self, *args, **kwargs):
    pass

  def build_instance(self, nn_framework):
    self.framework = nn_framework
    self.Y2node = dict()
    for n, y in nn_framework.node2Y.items():
      self.Y2node[y] = self.Y2node.get(y, []) + [n]
    for _from, _to in self.framework.edges:
      if _from not in self.node2X:
        self.node2X[_from] = randint(0,100)
      if _to not in self.node2X:
        self.node2X[_to] = randint(0,100)

  def train_instance(self, nn_framework) -> Dict:
    return dict()

  def mutate(self, prob):
    result = AncestryLayoutIndividual.__new__(AncestryLayoutIndividual)
    result.attr = self.attr
    result.node2X = {key: value for key, value in self.node2X.items()}
    result.edges = self.framework.edges
    result.Y2node = self.Y2node
    if random() < .5:
      for nodes in result.Y2node.values():
        if random() < prob and len(nodes) > 1:
          n0, n1 = sample(nodes, k=2)
          result.node2X[n0], result.node2X[n1] = result.node2X[n1], result.node2X[n0]
    else:
      for n, x in result.node2X.items():
        if random() < prob:
          result.node2X[n] = result.node2X[n] + round((random() - .5) / .25) * .5
    result._id_name = self.getNewName()
    return [result]

  def step(self, step_size):
    return self.mutate(step_size)

  def recombine(self, other):
    result = AncestryLayoutIndividual.__new__(AncestryLayoutIndividual)
    result.attr = self.attr
    result.framework = self.framework
    result.Y2node = self.Y2node

    result.node2X = dict()
    for n in self.node2X.keys():
      if random() < .5:
        result.node2X[n] = self.node2X[n] + \
                           round((other.node2X[n] - self.node2X[n]) * 2) / 2
      else:
        if random() < .5:
          result.node2X[n] = self.node2X[n]
        else:
          result.node2X[n] = other.node2X[n]
    result._id_name = self.getNewName()
    return [result]

  def layoutCrossingEdges(self):
    crossings = 1
    edges = list(self.framework.edges.keys())
    for i in range(len(edges)):
      ei0, ei1 = edges[i]
      ei0x, ei1x = self.node2X[ei0], self.node2X[ei1]
      iy = self.framework.node2Y[ei0], self.framework.node2Y[ei1]
      for j in range(len(edges)):
        ej0, ej1 = edges[j]
        ej0x, ej1x = self.node2X[ej0], self.node2X[ej1]
        jy = self.framework.node2Y[ej0], self.framework.node2Y[ej1]
        if (iy == jy and
            ((ei0x == ej0x and ei1x == ej1x) or
             ((ei0x < ej0x) != (ei1x < ej1x) and
              ei0x != ej0x and
              ei1x != ej0x))):
          crossings += 1
    return 1 / (crossings)

  def layoutDistanceX(self):
    d = 0
    for nodes in self.Y2node.values():
      for i in range(len(nodes)):
        ni = nodes[i]
        for j in range(i + 1, len(nodes)):
          nj = nodes[j]
          if ni not in self.framework.realNodes or nj not in self.framework.realNodes:
            t_d = .45
          else:
            t_d = 1
          dist = abs(self.node2X[ni] - self.node2X[nj]) / t_d
          if dist < 1:
            d += (2 / (1 + dist ** 2))
    return 1 / (1 + d)

  def layoutDistanceY(self):
    d = list()
    for e0, e1 in self.framework.edges:
      dist = abs(self.node2X[e0] - self.node2X[e1])
      if (e0 in self.framework.realNodes) and (e1 in self.framework.realNodes):
        d.append(dist ** 2)
      elif dist > .55:
        d.append((dist - .55) ** 2)
      else:
        d.append(dist ** 2 * .5)
    return 1 / (1 + sum(d) / len(d))

  @property
  def fitness(self):
    if hasattr(self, '_fitness') and self._fitness:
      return self._fitness
    elif self.metrics:
      self._fitness = 1
      for v in self.metrics.values():
        self._fitness *= v
      return self._fitness
    else:
      return -.1


class AncestryFramework(NeuralNetworkFrameworkInterface):
  def __init__(self, **kwargs):
    self._edges = dict()
    self._nodes2Y = dict()
    self._real_nodes = set()
    self.pseudo_node_idx = 0

  def add_edge(self, _from, _to, label):
    if (_from, _to) in self._edges or _from == _to:
      return False
    self._real_nodes.add(_from)
    self._real_nodes.add(_to)
    self._edges[_from, _to] = label
    Y_from = self._nodes2Y.get(_from)
    Y_to = self._nodes2Y.get(_to)
    if Y_from is None and Y_to is None:
      Y_from = 0
      Y_to = 1
    elif Y_from is None and Y_to is not None:
      Y_from = Y_to - 1
    elif Y_from is not None and Y_to is None:
      Y_to = Y_from + 1
    self._nodes2Y[_from] = Y_from
    self._nodes2Y[_to] = Y_to
    if Y_from == (Y_to - 1):
      return True
    if Y_to <= Y_from:
      self.resolveNodes(_from)
    self.resolveConnection()
    return True

  def resolveNodes(self, node):
    n2n = dict()
    for _from, _to in self.edges:
      n2n[_from] = n2n.get(_from, []) + [_to]
    stack = [node]
    while stack:
      node = stack.pop(0)
      Y = self._nodes2Y[node]
      for n in n2n.get(node, []):
        self._nodes2Y[n] = Y + 1
        stack.append(n)

  def resolveConnection(self):
    newEdges = dict()
    for (_from, _to), label in self._edges.items():
      if self._nodes2Y[_from] == (self._nodes2Y[_to] - 1):
        newEdges[_from, _to] = label
        continue
      _from_d = self._nodes2Y[_from]
      n_from = _from
      for i in range(1, self._nodes2Y[_to] - _from_d):
        n_to = 'dn_%i' % self.pseudo_node_idx
        self.pseudo_node_idx += 1
        self._nodes2Y[n_to] = _from_d + i
        newEdges[n_from, n_to] = label
        n_from = n_to
      newEdges[n_from, _to] = label
    self._edges = newEdges

  def reset_framework(self):
    pass

  def init_model(self, dataset_input_data: Set[str], dataset_target_data: Set[str]):
    pass

  def finalize_model(self, output_ids: List[Tuple[str, str]]):
    pass

  def set_weights(self, weights: Dict):
    pass

  def set_train_parameters(self, **kwargs):
    pass

  def add_function(self, function: Function):
    pass

  def train(self) -> Dict:
    return dict()

  def reset(self):
    pass

  def setup_individual(self, individual):
    pass

  @property
  def edges(self):
    return self._edges

  @property
  def node2Y(self):
    return self._nodes2Y

  @property
  def realNodes(self):
    return self._real_nodes
