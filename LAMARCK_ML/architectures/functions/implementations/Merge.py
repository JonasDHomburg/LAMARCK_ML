import math
from random import sample
from typing import Tuple, List, Dict, Set

from LAMARCK_ML.architectures.functions.interface import Function
from LAMARCK_ML.data_util import DimNames, TypeShape, Shape, IOLabel, \
  DHalf, \
  DFloat, \
  DDouble, \
  DInt64, \
  DInt32, \
  DInt16, \
  DInt8
from LAMARCK_ML.metrics.implementations import Parameters, FlOps


class Merge(Function,
            FlOps.Interface,
            Parameters.Interface):
  # IOLabel.MERGE_OTHER = 'MERGE_OTHER'
  IOLabel.MERGE_OTHER = 'DATA_SECOND'
  # IOLabel.MERGE_IN = 'MERGE_IN'
  IOLabel.MERGE_IN = 'DATA_FIRST'
  # IOLabel.MERGE_OUT = 'MERGE_OUT'
  IOLabel.MERGE_OUT = 'DATA_OUT'
  arg_OUT_NAMED_TYPE_SHAPES = 'outTypeShape'

  INPUTS = -1
  allowedTypes = [DDouble, DFloat, DHalf, DInt8, DInt16, DInt32, DInt64]
  _DF_INPUTS = [IOLabel.MERGE_IN, IOLabel.MERGE_OTHER]

  __min_f = .5
  __max_f = 1.

  @classmethod
  def possible_output_shapes(cls,
                             input_ntss: Dict[str, TypeShape],
                             target_output: TypeShape,
                             is_reachable,
                             max_possibilities: int = 10) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:

    target_shape = target_output.shape

    for label, nts in input_ntss.items():
      if nts.dtype not in cls.allowedTypes or \
          nts.dtype != target_output.dtype:
        continue

      possible_sizes = []
      names = []
      invalid_dim = False
      for _dim in nts.shape.dim:
        target_size = target_shape[_dim.name]
        if _dim.name == DimNames.WIDTH or \
            _dim.name == DimNames.HEIGHT or \
            _dim.name == DimNames.BATCH or \
            _dim.name == DimNames.TIME:
          pool = [_dim.size]
        elif _dim.name == DimNames.CHANNEL or \
            _dim.name == DimNames.UNITS:
          lower_border = max(math.floor(_dim.size * cls.__min_f), (min(2, target_size)
                                                                   if target_size is not None else 2))
          upper_border = math.ceil(_dim.size * cls.__max_f)
          pool = list(range(lower_border + 1, upper_border))
          border_pool = list({lower_border, upper_border})
          if target_size is None or not (lower_border < (target_size - _dim.size) < upper_border):
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool), 0), len(pool)))
          else:
            pool.remove(target_size - _dim.size)
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool) - 1, 0), len(pool))) + [
              target_size - _dim.size]
          pool = pool + border_pool
        else:
          invalid_dim = True
          break
        possible_sizes.append(pool)
        names.append(_dim.name)
      if invalid_dim:
        continue

      for dim_combination in Shape.random_dimension_product(possible_sizes):
        remaining_shape = Shape(*zip(names, dim_combination))
        out_nts = TypeShape(nts.dtype, Shape(*zip(names, [
          d_.size + _d.size if _d.name == DimNames.CHANNEL or _d.name == DimNames.UNITS else _d.size for d_, _d in
          zip(remaining_shape.dim, nts.shape.dim)])))
        if is_reachable(out_nts, target_output):
          yield ({IOLabel.MERGE_OTHER: TypeShape(nts.dtype, remaining_shape)},
                 {IOLabel.MERGE_OUT: out_nts},
                 {IOLabel.MERGE_IN: label})

  @classmethod
  def generateParameters(cls,
                         input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Set[TypeShape],
                         variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    default_input_label, default_input_ntss, _ = input_dict.get(IOLabel.MERGE_IN)
    other_input_label, other_input_ntss, _ = input_dict.get(IOLabel.MERGE_OTHER)
    default_input_nts = default_input_ntss[default_input_label]
    other_input_nts = other_input_ntss[other_input_label]

    if default_input_nts.dtype != other_input_nts.dtype:
      return [], []

    out_shape = Shape()

    for _dim in default_input_nts.shape.dim:
      if _dim.name == DimNames.CHANNEL or \
          _dim.name == DimNames.UNITS:
        out_shape.dim.append(Shape.Dim(_dim.name, _dim.size + other_input_nts.shape[_dim.name]))
      elif _dim.size != other_input_nts.shape[_dim.name]:
        return [], []
      else:
        out_shape.dim.append(Shape.Dim(_dim.name, _dim.size))

    input_mapping = dict([(l_in, (l_out, df_id_name)) for l_in, (l_out, _, df_id_name) in input_dict.items()])
    return [{cls.arg_INPUT_MAPPING: input_mapping,
             cls.arg_ATTRIBUTES: {cls.arg_OUT_NAMED_TYPE_SHAPES:
                                    {IOLabel.MERGE_OUT: TypeShape(default_input_nts.dtype, out_shape)}},
             }], [1.0]

  def __init__(self, **kwargs):
    super(Merge, self).__init__(**kwargs)
    if not (isinstance(self.attr[self.arg_OUT_NAMED_TYPE_SHAPES], dict) and
            all([isinstance(nts, TypeShape) and isinstance(label, str) for label, nts in
                 self.attr[self.arg_OUT_NAMED_TYPE_SHAPES].items()])):
      raise Exception('Wrong output TypeShapes!')

  @property
  def outputs(self) -> Set[TypeShape]:
    return self.attr[self.arg_OUT_NAMED_TYPE_SHAPES]

  @classmethod
  def min_transform(cls, nts):
    if nts.dtype not in cls.allowedTypes:
      return None
    s = Shape()
    image, vektor = False, False
    result = TypeShape(nts.dtype, s)
    for _dim in nts.shape.dim:
      if (_dim.name == DimNames.BATCH
          or _dim.name == DimNames.WIDTH
          or _dim.name == DimNames.HEIGHT
      ):
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif _dim.name == DimNames.UNITS and not image:
        s.dim.append(Shape.Dim(_dim.name, int(math.floor(_dim.size * (1 + cls.__min_f)))))
        vektor = True
      elif _dim.name == DimNames.CHANNEL and not vektor:
        s.dim.append(Shape.Dim(_dim.name, int(math.floor(_dim.size * (1 + cls.__min_f)))))
        image = True
      else:
        return None
    return result

  @classmethod
  def max_transform(cls, nts):
    if nts.dtype not in cls.allowedTypes:
      return None
    s = Shape()
    image, vektor = False, False
    result = TypeShape(nts.dtype, s)
    for _dim in nts.shape.dim:
      if (_dim.name == DimNames.BATCH
          or _dim.name == DimNames.WIDTH
          or _dim.name == DimNames.HEIGHT
      ):
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif _dim.name == DimNames.UNITS and not image:
        s.dim.append(Shape.Dim(_dim.name, int(math.ceil(_dim.size * (1 + cls.__max_f)))))  # pessimistic estimation
        vektor = True
      elif _dim.name == DimNames.CHANNEL and not vektor:
        s.dim.append(Shape.Dim(_dim.name, int(math.ceil(_dim.size * (1 + cls.__max_f)))))  # pessimistic estimation
        image = True
      else:
        return None
    return result

  def flops_per_sample(self):
    return 0

  def parameters(self):
    return 0