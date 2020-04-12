from typing import List, Dict, Tuple

from LAMARCK_ML.architectures.functions.interface import Function
from LAMARCK_ML.data_util import IOLabel, TypeShape

from LAMARCK_ML.data_util.dataType import \
  DHalf, \
  DFloat, \
  DDouble, \
  DInt64, \
  DInt32, \
  DInt16, \
  DInt8


class Softmax(Function,
              ):
  allowedTypes = {DFloat, DDouble, DHalf, DInt8, DInt16, DInt32, DInt64}
  IOLabel.SOFTMAX_OUT = 'SOFTMAX_OUT'
  IOLabel.SOFTMAX_IN = 'SOFTMAX_IN'
  arg_OUT_TYPE_SHAPE = 'outTypeShape'

  @classmethod
  def possible_output_shapes(cls, input_ntss: Dict[str, TypeShape], target_output: TypeShape, is_reachable,
                             max_possibilities: int = 10, **kwargs) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:
    for label, nts in input_ntss.items():
      if nts.dtype not in cls.allowedTypes:
        continue
      yield ({},
             {IOLabel.SOFTMAX_OUT: nts},
             {IOLabel.SOFTMAX_IN: label})

  @classmethod
  def generateParameters(cls, input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Dict[str, TypeShape], variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    if len(input_dict) != 1 or \
        len(expected_outputs) != 1:
      print(len(input_dict))
      print(len(expected_outputs))
      return [], []

    return [{cls.arg_ATTRIBUTES: {cls.arg_OUT_TYPE_SHAPE: expected_outputs, },
             cls.arg_VARIABLES: [],
             cls.arg_INPUT_MAPPING: {l_in: (l_out, id_name) for l_in, (l_out, _, id_name) in input_dict.items()}
             }], [1]

  @classmethod
  def min_transform(cls, nts: TypeShape):
    if nts.dtype not in cls.allowedTypes:
      return None
    return nts

  @classmethod
  def max_transform(cls, nts: TypeShape):
    if nts.dtype not in cls.allowedTypes:
      return None
    return nts

  def __init__(self, **kwargs):
    super(Softmax, self).__init__(**kwargs)
    if not (isinstance(self.attr[self.arg_OUT_TYPE_SHAPE], dict) and
            all([isinstance(nts, TypeShape) and isinstance(label, str) for label, nts in
                 self.attr[self.arg_OUT_TYPE_SHAPE].items()])):
      raise Exception('Wrong output TypeShapes!')

  @property
  def outputs(self) -> Dict[str, TypeShape]:
    return self.attr[self.arg_OUT_TYPE_SHAPE]

  @property
  def inputLabels(self) -> List[str]:
    return list(self.input_mapping.keys())
