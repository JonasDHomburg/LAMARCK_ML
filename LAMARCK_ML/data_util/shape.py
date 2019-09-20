from enum import Enum, unique

import numpy as np

from LAMARCK_ML.data_util.Shape_pb2 import ShapeProto


class InvalidDimension(Exception):
  pass


class Shape(object):
  @staticmethod
  def random_dimension_product(samples, target=None):
    option_count = list()
    counter = 1
    for s in samples[::-1]:
      option_count.append(counter)
      counter *= len(s)
    option_count = option_count[::-1]

    probabilities_dict = dict()
    index_lists = [len(s) for s in samples]

    probabilities_dict[tuple()] = [option_count[0] for _ in samples[0]]
    if target is not None:
      for idx in range(len(target)):
        try:
          op_idx = samples[idx].index(target[idx])
          key = tuple(target[:idx])
          c = probabilities_dict.get(key)
          if c is None:
            c = [option_count[idx] for _ in samples[idx]]
          c[op_idx] -= 1
          probabilities_dict[key] = c
        except:
          raise Exception('target not in options')
    samples_left = sum(probabilities_dict.get(tuple(), [0])) > 0
    sample_depth = len(samples)
    while samples_left:
      current_sample = []
      for depth in range(sample_depth):
        current_as_tuple = tuple(current_sample)
        pool = index_lists[depth]
        c = probabilities_dict.get(current_as_tuple)
        if c is None:
          c = [option_count[depth] for _ in samples[depth]]
        p = np.asarray(c)
        p = p / np.sum(p)
        s = np.random.choice(pool, size=1, replace=False, p=p)[0]

        c[s] -= 1
        probabilities_dict[current_as_tuple] = c
        current_sample.append(samples[depth][s])
        if depth == 0:
          samples_left = sum(c) > 0
      yield current_sample

  class Dim(object):
    @unique
    class Names(Enum):
      BATCH = 'N'
      CHANNEL = 'C'
      WIDTH = 'W'
      HEIGHT = 'H'
      UNITS = 'U'
      TIME = 'T'

    def __init__(self, *args, **kwargs):
      self.name = kwargs.get('name')
      if self.name is None:
        self.name = args[0]
      if not isinstance(self.name, Shape.Dim.Names):
        self.name = Shape.Dim.Names(self.name)
      self.size = kwargs.get('size')
      if self.size is None:
        self.size = args[1] if len(args) > 1 else args[0]
      self.size = self.size
      pass

    def __str__(self):
      return str(self.name.value) + ': ' + str(self.size)

    def __eq__(self, other):
      if not isinstance(other, self.__class__) or \
          not self.size == other.size or \
          not self.name == other.name:
        return False
      return True

    def __hash__(self):
      return hash(int.from_bytes(self.name.value.encode('utf-8'), byteorder='big') * 13 + hash(self.size))

    def __ne__(self, other):
      return not self.__eq__(other)

    def __copy__(self):
      return Shape.Dim(self.name, self.size)

  def __init__(self, *args):
    '''
    Careful! Order matters!!
    Depending on the order the data gets transposed for the functions!
    '''
    self.dim = []
    super(Shape, self).__init__()
    for arg in args:
      _str, _num = arg
      try:
        self.dim.append(Shape.Dim(_str, _num))
      except Exception as e:
        pass

  def get_pb(self, result=None):
    if not isinstance(result, ShapeProto):
      result = ShapeProto()
    for _dim in self.dim:
      if _dim.name is not None:
        dim_ = ShapeProto.Dim()
        dim_.name = _dim.name.value
        dim_.size = 0 if _dim.size is None else _dim.size
        result.dim.append(dim_)
    return result

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def __setstate__(self, state):
    if isinstance(state, str) or isinstance(state, bytes):
      _shape = ShapeProto()
      _shape.ParseFromString(state)
    elif isinstance(state, ShapeProto):
      _shape = state
    else:
      return
    self.dim = list()
    for _dim in _shape.dim:
      self.dim.append(Shape.Dim(_dim.name, _dim.size if _dim.size != 0 else None))

  def __str__(self):
    result = "[" + "; ".join([str(_dim) for _dim in self.dim]) + "]"
    return result

  def __eq__(self, other):
    if isinstance(other, self.__class__) and \
        len(self.dim) == len(other.dim) and \
        not any([not _self == _other for _self, _other in zip(self.dim, other.dim)]):
      return True
    return False

  def __hash__(self):
    result = 0
    for d in self.dim:
      result = hash(result * 13 + hash(d))
    return result

  def __copy__(self):
    result = Shape()
    for _dim in self.dim:
      result.dim.append(_dim.__copy__())
    return result

  def __getitem__(self, item):
    for _dim in self.dim:
      if _dim.name == item:
        return _dim.size

  @property
  def as_dict(self):
    result = dict()
    for d in self.dim:
      result[d.name] = d.size
    return result

  @property
  def units(self):
    _units = 1
    for dim in self.dim:
      if (dim.name != self.Dim.Names.BATCH
          and dim.name != self.Dim.Names.TIME
      ):
        _units = _units * dim.size
    return _units

  def __cmp__(self, other):
    if (not isinstance(other, self.__class__)
        or len(self.dim) != len(other.dim)):
      return 0
    self_d = self.as_dict
    other_d = other.as_dict
    if not (len(set(self_d.keys()).union(set(other_d.keys()))) ==
            len(self_d) == len(other_d)):
      return 0

    cmp = [self_d[key] <= other_d[key] for key in self_d.keys() if key != Shape.Dim.Names.BATCH]
    if all(cmp):
      return -1
    if not any(cmp):
      return 1
    return 0
