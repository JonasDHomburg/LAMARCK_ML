from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.reproduction.methods import Mutation, Recombination
from LAMARCK_ML.metrics import LayoutDistanceX, LayoutDistanceY, LayoutCrossingEdges

from random import random, sample, seed

seed()


class GraphLayoutIndividual(IndividualInterface, Mutation.Interface, Recombination.Interface,
                            LayoutCrossingEdges.Interface, LayoutDistanceX.Interface,
                            LayoutDistanceY.Interface):
  arg_EDGES = 'edges'
  arg_DISTANCE = 'distance'
  arg_METRIC_WEIGHTS = 'metric_weights'
  attr_NODES = 'nodes'
  attr_REAL_NODES = 'real_nodes'
  attr_D2N = 'd2n'
  attr_D2E = 'd2e'
  attr_N2X = 'n2x'

  def __init__(self, **kwargs):
    super(GraphLayoutIndividual, self).__init__(**kwargs)
    self.distance = kwargs.get(self.arg_DISTANCE, 1)
    self.edges = set(kwargs.get(self.arg_EDGES))
    self.metric_weights = kwargs.get(self.arg_METRIC_WEIGHTS, dict())
    if not all([isinstance(e, tuple) for e in self.edges]):
      raise Exception('Expected set or list of tuples for ' + self.arg_EDGES)
    self.nodes = set()

    self.depth2nodes = dict()
    self.depth2edges = dict()
    self.real_nodes = set()

    n2subn = dict()
    for e0, e1 in self.edges:
      n2subn[e1] = n2subn.get(e1, []) + [e0]
      self.real_nodes.add(e0)
      self.real_nodes.add(e1)

    def n2depth(n, subn, mem=dict()):
      stack = [n]
      while stack:
        current = stack[-1]
        if current not in subn:
          mem[current] = 0
          stack.pop(-1)
          continue
        if current not in mem:
          max_d = -1
          for _n in subn[current]:
            if _n not in mem:
              stack.append(_n)
            else:
              max_d = max(max_d, mem[_n])
          if max_d > -1:
            mem[current] = max_d +1
            stack.pop(-1)
        else:
          stack.pop(-1)
      return mem[n]

    n2d = dict()
    for _, e1 in self.edges:
      n2depth(e1, n2subn, n2d)

    for n, d in n2d.items():
      self.depth2nodes[d] = self.depth2nodes.get(d, []) + [n]

    tmp_idx = 0
    new_edges = set()
    for e0, e1 in self.edges:
      self.nodes.add(e0)
      self.nodes.add(e1)
      if n2d[e1] - n2d[e0] > 1:
        n_ = e0
        e0d = n2d[e0]
        for i in range(1, n2d[e1] - n2d[e0]):
          _n = 'dn_%i' % tmp_idx
          tmp_idx += 1
          self.nodes.add(_n)
          e = (n_, _n)
          new_edges.add(e)
          self.depth2edges[e0d + i - 1] = self.depth2edges.get(e0d + i - 1, []) + [e]
          self.depth2nodes[e0d + i] = self.depth2nodes.get(e0d + i, []) + [_n]
          n_ = _n
        e = (n_, e1)
        new_edges.add(e)
        self.depth2edges[e0d + i] = self.depth2edges.get(e0d + i, []) + [e]
      else:
        e = (e0, e1)
        self.depth2edges[n2d[e0]] = self.depth2edges.get(n2d[e0], []) + [e]
        new_edges.add(e)
    self.edges = new_edges

    self.node2X = dict()
    for nodes in self.depth2nodes.values():
      k = len(nodes)
      for n, x in zip(nodes, sample(list(range(k)), k=k)):
        self.node2X[n] = x * self.distance

    self.attr[self.attr_D2E] = list(self.depth2edges.items())
    self.attr[self.attr_D2N] = list(self.depth2nodes.items())
    self.attr[self.attr_NODES] = self.nodes
    self.attr[self.attr_REAL_NODES] = self.real_nodes
    self.attr[self.arg_EDGES] = self.edges
    self.attr[self.attr_N2X] = list(self.node2X.items())
    self.attr[self.arg_DISTANCE] = self.distance
    self.attr[self.arg_METRIC_WEIGHTS] = list(self.metric_weights.items())

  def __sub__(self, other):
    distance = 0
    for k in self.nodes:
      distance += (self.node2X[k] - other.node2X[k]) ** 2
    return distance

  def norm(self, other):
    return self - other

  def _cls_setstate(self, state):
    super(GraphLayoutIndividual, self)._cls_setstate(state)
    self.depth2edges = dict(self.attr[self.attr_D2E])
    self.depth2nodes = dict(self.attr[self.attr_D2N])
    self.nodes = self.attr[self.attr_NODES]
    self.real_nodes = self.attr[self.attr_REAL_NODES]
    self.edges = self.attr[self.arg_EDGES]
    self.node2X = dict(self.attr[self.attr_N2X])
    self.distance = self.attr[self.arg_DISTANCE]
    self.metric_weights = dict(self.attr[self.arg_METRIC_WEIGHTS])

  def update_state(self, *args, **kwargs):
    pass

  def layoutCrossingEdges(self):
    crossings = 1
    for edges in self.depth2edges.values():
      for i in range(len(edges)):
        ei0, ei1 = edges[i]
        for j in range(i + 1, len(edges)):
          ej0, ej1 = edges[j]
          if ((self.node2X[ei0] < self.node2X[ej0]) !=
              (self.node2X[ei1] < self.node2X[ej1]) and
              self.node2X[ei0] != self.node2X[ej0] and
              self.node2X[ei1] != self.node2X[ej1]):
            crossings += 1
    return 1 / (crossings)

  def layoutDistanceX(self):
    d = 0
    for nodes in self.depth2nodes.values():
      for i in range(len(nodes)):
        ni = nodes[i]
        for j in range(i + 1, len(nodes)):
          nj = nodes[j]
          if ni not in self.real_nodes or nj not in self.real_nodes:
            t_d = self.distance * .45
          else:
            t_d = self.distance
          dist = abs(self.node2X[ni] - self.node2X[nj]) / t_d
          if dist < 1:
            d += (2 / (1 + dist ** 2))
    return 1 / (1 + d)

  def layoutDistanceY(self):
    d = list()
    for e0, e1 in self.edges:
      dist = abs(self.node2X[e0] - self.node2X[e1]) / self.distance
      if (e0 in self.real_nodes) and (e1 in self.real_nodes):
        d.append(dist ** 2)
      elif dist > .55:
        d.append((dist - .55) ** 2)
      else:
        d.append(dist ** 2 * .5)
    return 1 / (1 + sum(d) / len(d))

  def mutate(self, prob):
    result = GraphLayoutIndividual.__new__(GraphLayoutIndividual)
    result.attr = self.attr
    result.depth2edges = self.depth2edges
    result.depth2nodes = self.depth2nodes
    result.nodes = self.nodes
    result.real_nodes = self.real_nodes
    result.edges = self.edges
    result.distance = self.distance
    result.node2X = self.node2X
    result.metric_weights = self.metric_weights
    if random() < .5:
      for nodes in result.depth2nodes.values():
        if random() < prob and len(nodes) > 1:
          n0, n1 = sample(nodes, k=2)
          result.node2X[n0], result.node2X[n1] = result.node2X[n1], result.node2X[n0]
    else:
      for n, x in result.node2X.items():
        if random() < prob:
          result.node2X[n] = result.node2X[n] + round((random() - .5) / .25) * .5 * self.distance
    result._id_name = self.getNewName()
    return [result]

  def recombine(self, other):
    result = GraphLayoutIndividual.__new__(GraphLayoutIndividual)
    result.attr = self.attr
    result.depth2edges = self.depth2edges
    result.depth2nodes = self.depth2nodes
    result.nodes = self.nodes
    result.real_nodes = self.real_nodes
    result.edges = self.edges
    result.distance = self.distance
    result.metric_weights = self.metric_weights

    result.node2X = dict()
    for n in self.node2X.keys():
      if random() < .5:
        result.node2X[n] = self.node2X[n] + \
                           round((other.node2X[n] - self.node2X[n]) * 2 / self.distance) * self.distance / 2
      else:
        if random() < .5:
          result.node2X[n] = self.node2X[n]
        else:
          result.node2X[n] = other.node2X[n]
    result.attr[self.attr_N2X] = list(result.node2X.items())
    result._id_name = self.getNewName()
    return [result]

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

  def build_instance(self, nn_framework):
    pass

  def train_instance(self, nn_framework):
    return dict()
