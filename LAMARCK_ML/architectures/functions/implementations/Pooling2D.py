import math
from enum import Enum
from random import sample
from typing import Tuple, List, Dict, Set
from random import random, choice

from LAMARCK_ML.architectures.functions.interface import Function
from LAMARCK_ML.data_util import DimNames, TypeShape, IOLabel, Shape
from LAMARCK_ML.data_util.dataType import \
  DHalf, \
  DFloat, \
  DDouble, \
  DInt64, \
  DInt32, \
  DInt16, \
  DInt8, \
  DUInt8, \
  DUInt16, \
  DUInt32, \
  DUInt64
from LAMARCK_ML.reproduction.methods import Mutation


class Pooling2D(Function, Mutation.Interface):
  # IOLabel.POOLING2D_IN = 'POOLING2D_IN'
  IOLabel.POOLING2D_IN = 'DATA_IN'
  # IOLabel.POOLING2D_OUT = 'POOLING2D_OUT'
  IOLabel.POOLING2D_OUT = 'DATA_OUT'

  class Padding(Enum):
    SAME = 'SAME'
    VALID = 'VALID'

  class PoolingType(Enum):
    MIN = 'MIN'
    MAX = 'MAX'
    MEAN = 'MEAN'

  allowedTypes = [DHalf, DFloat, DDouble, DInt8, DInt16, DInt32, DInt64, DUInt8, DUInt16, DUInt32, DUInt64]
  _DF_INPUTS = [IOLabel.POOLING2D_IN]

  arg_OUT_NAMED_TYPE_SHAPES = 'outTypeShape'
  arg_POOLING_WIDTH = 'pooling_width'
  arg_POOLING_HEIGHT = 'pooling_height'
  arg_STRIDE_WIDTH = 'stride_width'
  arg_STRIDE_HEIGHT = 'stride_height'
  arg_IN_WIDTH = 'in_width'
  arg_IN_HEIGHT = 'in_height'
  arg_PADDING = 'padding'
  arg_POOLING_TYPE = 'pooling_type'

  __min_f_hw = .5
  __max_f_hw = 1

  @classmethod
  def possible_output_shapes(cls,
                             input_ntss: Dict[str, TypeShape],
                             target_output: TypeShape,
                             is_reachable,
                             max_possibilities: int = 10) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:

    target_shape = target_output.shape

    for label, nts in input_ntss.items():
      if nts.dtype not in cls.allowedTypes:
        continue
      possible_sizes = []
      names = []
      invalid_dim = False
      for _dim in nts.shape.dim:
        target_size = target_shape[_dim.name]
        if _dim.name == DimNames.WIDTH or \
            _dim.name == DimNames.HEIGHT:
          lower_border = max(math.floor(_dim.size * cls.__min_f_hw), (min(2, target_size)
                                                                      if target_size is not None else 2))
          upper_border = math.ceil(_dim.size * cls.__max_f_hw)
          pool = list(range(lower_border + 1, upper_border))
          border_pool = list({upper_border, lower_border})
          if target_size is None or not (lower_border < target_size < upper_border):
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool), 0), len(pool)))
          else:
            pool.remove(target_size)
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool) - 1, 0), len(pool))) + [target_size]
          pool = pool + border_pool
        elif _dim.name == DimNames.CHANNEL or \
            _dim.name == DimNames.BATCH:
          pool = [_dim.size]
        else:
          invalid_dim = True
          break
        possible_sizes.append(pool)
        names.append(_dim.name)
      if invalid_dim:
        continue

      for comb in Shape.random_dimension_product(possible_sizes):
        out_nts = TypeShape(nts.dtype, Shape(*zip(names, comb)))
        if is_reachable(out_nts, target_output):
          yield ({},
                 {IOLabel.POOLING2D_OUT: out_nts},
                 {IOLabel.POOLING2D_IN: label})

  @classmethod
  def configurations(cls, h_i, h_o, w_i, w_o):
    configurations = list()

    def stride_range(in_, out_):
      if out_ == 1:
        return [in_]
      else:
        lower_limit = math.ceil(in_ / out_)
        upper_limit = math.ceil(in_ / (out_ - 1)) - 1
        if lower_limit == 0 or \
            math.ceil(in_ / lower_limit) < out_ or \
            upper_limit == 0 or \
            math.ceil(in_ / upper_limit) > out_:
          return []
        return list(range(lower_limit, upper_limit + 1))

    def pooling_range(in_, out_):
      lower_limit = 1
      upper_limit = in_ - out_ + 1
      return list(range(lower_limit, upper_limit + 1))

    for s_h in stride_range(h_i, h_o):
      for s_w in stride_range(w_i, w_o):
        for p_h in range(1, 7):
          for p_w in range(1, 7):
            for _t in Pooling2D.PoolingType:
              configurations.append({
                cls.arg_POOLING_HEIGHT: p_h,
                cls.arg_POOLING_WIDTH: p_w,
                cls.arg_STRIDE_HEIGHT: s_h,
                cls.arg_STRIDE_WIDTH: s_w,
                cls.arg_PADDING: Pooling2D.Padding.SAME.value,
                cls.arg_POOLING_TYPE: _t.value,
              })
    for p_w in pooling_range(w_i, w_o):
      for p_h in pooling_range(h_i, h_o):
        for s_w in stride_range(w_i - p_w + 1, w_o):
          for s_h in stride_range(h_i - p_h + 1, h_o):
            for _t in Pooling2D.PoolingType:
              configurations.append({
                cls.arg_POOLING_HEIGHT: p_h,
                cls.arg_POOLING_WIDTH: p_w,
                cls.arg_STRIDE_HEIGHT: s_h,
                cls.arg_STRIDE_WIDTH: s_w,
                cls.arg_PADDING: Pooling2D.Padding.VALID.value,
                cls.arg_POOLING_TYPE: _t.value,
              })
    return configurations

  @classmethod
  def generateParameters(cls,
                         input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Dict[str, TypeShape],
                         variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    if len(input_dict) != 1 or \
        len(expected_outputs) != 1:
      return [], []

    input_nts_id, input_outputs, input_id = input_dict[IOLabel.POOLING2D_IN]
    in_nts = input_outputs[input_nts_id]
    out_label, out_nts = next(iter(expected_outputs.items()))
    if in_nts.dtype != out_nts.dtype:
      return [], []

    allowed_dimensions = {DimNames.BATCH, DimNames.CHANNEL, DimNames.WIDTH, DimNames.HEIGHT}
    for _dim in in_nts.shape.dim:
      if _dim.name not in allowed_dimensions:
        return [], []
    for _dim in out_nts.shape.dim:
      if _dim.name not in allowed_dimensions:
        return [], []

    h_i = in_nts.shape[DimNames.HEIGHT]
    h_o = out_nts.shape[DimNames.HEIGHT]
    w_i = in_nts.shape[DimNames.WIDTH]
    w_o = out_nts.shape[DimNames.WIDTH]
    configurations = Pooling2D.configurations(h_i=h_i, h_o=h_o, w_i=w_i, w_o=w_o)

    results = [
      {cls.arg_ATTRIBUTES: {**config, **{cls.arg_OUT_NAMED_TYPE_SHAPES: {out_label: out_nts},
                                         cls.arg_IN_WIDTH: w_i,
                                         cls.arg_IN_HEIGHT: h_i}},
       cls.arg_INPUT_MAPPING: dict([(l_in, (l_out, id_name)) for l_in, (l_out, _, id_name) in input_dict.items()])}
      for config in configurations]
    prob = 1 / len(results)
    probabilities = [prob for _ in range(len(results))]
    return results, probabilities

  def __init__(self, **kwargs):
    super(Pooling2D, self).__init__(**kwargs)
    if not (isinstance(self.attr[self.arg_OUT_NAMED_TYPE_SHAPES], dict) and
            all([isinstance(nts, TypeShape) and isinstance(label, str) for label, nts in
                 self.attr[self.arg_OUT_NAMED_TYPE_SHAPES].items()])):
      raise Exception('Wrong output TypeShapes!')

  @property
  def outputs(self) -> Set[TypeShape]:
    return self.attr[self.arg_OUT_NAMED_TYPE_SHAPES]

  def mutate(self, prob):
    result = Pooling2D.__new__(Pooling2D)
    result.__setstate__(self.get_pb())
    if random() < prob:
      result._name = Pooling2D.getNewName()
      out_nts = self.attr[self.arg_OUT_NAMED_TYPE_SHAPES][IOLabel.POOLING2D_OUT]
      h_o = out_nts.shape[DimNames.HEIGHT]
      w_o = out_nts.shape[DimNames.WIDTH]
      result.attr = {**result.attr,
                     **choice(Pooling2D.configurations(
                       h_i=self.attr[self.arg_IN_HEIGHT],
                       h_o=h_o,
                       w_i=self.attr[self.arg_IN_WIDTH],
                       w_o=w_o
                     ))}
    return result

  @classmethod
  def min_transform(cls, nts):
    if nts.dtype not in cls.allowedTypes:
      return None
    s = Shape()
    result = TypeShape(nts.dtype, s)
    for _dim in nts.shape.dim:
      if _dim.name == DimNames.BATCH or \
          _dim.name == DimNames.CHANNEL:
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif _dim.name == DimNames.WIDTH or \
          _dim.name == DimNames.HEIGHT:
        s.dim.append(Shape.Dim(_dim.name, int(math.floor(_dim.size * cls.__min_f_hw))))
      else:
        return None
    return result

  @classmethod
  def max_transform(cls, nts):
    if nts.dtype not in cls.allowedTypes:
      return None
    s = Shape()
    result = TypeShape(nts.dtype, s)
    for _dim in nts.shape.dim:
      if _dim.name == DimNames.BATCH or \
          _dim.name == DimNames.CHANNEL:
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif _dim.name == DimNames.WIDTH or \
          _dim.name == DimNames.HEIGHT:
        s.dim.append(Shape.Dim(_dim.name, int(math.floor(_dim.size * cls.__max_f_hw))))
      else:
        return None
    return result

  pass
