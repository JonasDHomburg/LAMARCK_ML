import math
from typing import Tuple, List, Dict, Set

from LAMARCK_ML.architectures.functions.interface import Function
from LAMARCK_ML.data_util import Shape, DimNames, TypeShape, IOLabel
from LAMARCK_ML.data_util.dataType import *


class Flatten(Function):
  allowedTypes = [DHalf,
                  DFloat,
                  DDouble,
                  DInt8,
                  DInt16,
                  DInt32,
                  DInt64,
                  DBool,
                  DComplex64,
                  DComplex128,
                  DUInt8,
                  DUInt16,
                  DUInt32,
                  DUInt64,
                  ]
  _DF_INPUTS = [IOLabel.DEFAULT]

  arg_OUT_NAMED_TYPE_SHAPES = 'outTypeShape'
  # IOLabel.FLATTEN_IN = 'FLATTEN_IN'
  IOLabel.FLATTEN_IN = 'DATA_IN'
  # IOLabel.FLATTEN_OUT = 'FLATTEN_OUT'
  IOLabel.FLATTEN_OUT = 'DATA_OUT'

  @classmethod
  def possible_output_shapes(cls,
                             input_ntss: Dict[str, TypeShape],
                             target_output: TypeShape,
                             is_reachable,
                             max_possibilities: int = 10) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:

    allowed_in_dimensions = {DimNames.CHANNEL, DimNames.WIDTH, DimNames.HEIGHT}

    # target_shape = target_output.shape

    for label, nts in input_ntss.items():
      if nts.dtype not in cls.allowedTypes:
        continue
      units = 1
      invalid_dim = False
      batch = False
      batch_size = -1
      for _dim in nts.shape.dim:
        if _dim.name in allowed_in_dimensions:
          units *= _dim.size
        elif _dim.name == DimNames.BATCH:
          batch = True
          batch_size = _dim.size
        else:
          invalid_dim = True
          break
      if invalid_dim:
        continue
      out_nts = TypeShape(nts.dtype, Shape((DimNames.BATCH, batch_size),
                                                            (DimNames.UNITS, units))) if batch else \
        TypeShape(nts.dtype, Shape((DimNames.UNITS, units)))
      if is_reachable(out_nts, target_output):
        yield ({},
               {IOLabel.FLATTEN_OUT: out_nts},
               {IOLabel.FLATTEN_IN: label})

  @classmethod
  def generateParameters(cls,
                         input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Dict[str, TypeShape],
                         variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    input_nts_id, input_outputs, input_id = input_dict[IOLabel.FLATTEN_IN]
    in_nts = input_outputs[input_nts_id]
    out_label, out_nts = next(iter(expected_outputs.items()))
    if in_nts.dtype != out_nts.dtype:
      return [], []

    allowed_in_dimensions = {DimNames.BATCH, DimNames.CHANNEL, DimNames.WIDTH, DimNames.HEIGHT}
    for _dim in in_nts.shape.dim:
      if _dim.name not in allowed_in_dimensions:
        return [], []
    allowed_out_dimensions = {DimNames.BATCH, DimNames.UNITS}
    for _dim in out_nts.shape.dim:
      if _dim.name not in allowed_out_dimensions:
        return [], []

    return [{
      cls.arg_ATTRIBUTES: {cls.arg_OUT_NAMED_TYPE_SHAPES: {out_label: out_nts}},
      cls.arg_INPUT_MAPPING: dict([(l_in, (l_out, id_name)) for l_in, (l_out, _, id_name) in input_dict.items()]),
      cls.arg_VARIABLES: []}], \
           [1]

  def __init__(self, **kwargs):
    super(Flatten, self).__init__(**kwargs)
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
    result = TypeShape(nts.dtype, s)
    units = 1
    num_units = 0
    for _dim in nts.shape.dim:
      if _dim.name == DimNames.BATCH:
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif (_dim.name == DimNames.WIDTH or
            _dim.name == DimNames.HEIGHT or
            _dim.name == DimNames.CHANNEL):
        units *= _dim.size
      elif _dim.name == DimNames.UNITS:
        units *= _dim.size
        num_units += 1
      else:
        return None
    if num_units == 1:
      return None
    s.dim.append(Shape.Dim(DimNames.UNITS, units))
    return result

  @classmethod
  def max_transform(cls, nts):
    return cls.min_transform(nts)
