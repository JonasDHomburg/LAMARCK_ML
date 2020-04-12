from random import random, choice
from typing import Tuple, List, Dict, Set
import numpy as np

from LAMARCK_ML.architectures.functions.interface import Function
from LAMARCK_ML.architectures.functions.activations import Activations
from LAMARCK_ML.data_util import DimNames, TypeShape, IOLabel
from LAMARCK_ML.data_util.dataType import \
  DHalf, \
  DFloat, \
  DDouble, \
  DInt64, \
  DInt32, \
  DInt16, \
  DInt8, \
  DBinary
from LAMARCK_ML.data_util.shape import Shape
from LAMARCK_ML.metrics.implementations import FlOps, Parameters
from LAMARCK_ML.reproduction.methods import Mutation


class Perceptron(Function,
                 FlOps.Interface,
                 Parameters.Interface,
                 Mutation.Interface,
                 ):
  allowedTypes = {DFloat, DDouble, DHalf, DInt8, DInt16, DInt32, DInt64, DBinary}
  allowedActivations = [Activations.sigmoid, Activations.tanh,
                        Activations.linear, Activations.relu,
                        Activations.selu, Activations.elu,
                        Activations.exponential, Activations.hard_sigmoid,
                        Activations.softplus, Activations.softsign,
                        Activations.sign, Activations.sine,
                        Activations.cosine, Activations.absolute,
                        Activations.inverse, Activations.gaussian]
  IOLabel.PERCEPTRON_OUT = 'PERCEPTRON_OUT'
  PERCEPTRON_IN = 'PERCEPTRON_IN_'

  arg_OUT_TYPE_SHAPE = 'outTypeShape'

  @classmethod
  def get_in_label(cls, value: int):
    return cls.PERCEPTRON_IN + '{0:03.0f}'.format(value)

  @classmethod
  def possible_output_shapes(cls,
                             input_ntss: Dict[str, TypeShape],
                             target_output: TypeShape,
                             is_reachable,
                             max_possibilities: int = 10,
                             max_inputs: int = None,
                             **kwargs) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:
    for label, nts in input_ntss.items():
      if nts.dtype not in cls.allowedTypes:
        continue
      possible_sizes = []
      names = []
      invalid_dim = False
      for _dim in nts.shape.dim:
        if _dim.name == DimNames.UNITS and _dim.size == 1:
          pool = [1]
        elif _dim.name == DimNames.BATCH:
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
          yield ({cls.get_in_label(i + 1): nts.__copy__() for i in
                  range(round(np.random.beta(2, 5) * (10 if not isinstance(max_inputs, int) else max_inputs - 1)))},
                 {IOLabel.PERCEPTRON_OUT: out_nts},
                 {cls.get_in_label(0): label})
      break

  @classmethod
  def generateParameters(cls,
                         input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Dict[str, TypeShape],
                         variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    if len(input_dict) < 1 or \
        len(expected_outputs) != 1:
      return [], []

    input_nts_id, input_outputs, _ = input_dict[cls.get_in_label(0)]
    in_nts = input_outputs[input_nts_id]
    for input_nts_id, input_outputs, _ in input_dict.values():
      if input_outputs[input_nts_id] != in_nts:
        return [], []

    _, out_nts = next(iter(expected_outputs.items()))
    if in_nts.dtype != out_nts.dtype:
      return [], []

    allowed_dimensions = {DimNames.BATCH, DimNames.UNITS}
    for _dim in out_nts.shape.dim:
      if _dim.name not in allowed_dimensions:
        return [], []
      elif _dim.name == DimNames.UNITS and _dim.size != 1:
        return [], []

    prob = 1 / len(Perceptron.allowedActivations)
    return [{cls.arg_ATTRIBUTES: {
      cls.arg_OUT_TYPE_SHAPE: expected_outputs,
      cls.arg_ACTIVATION: a},
      cls.arg_INPUT_MAPPING: {l_in: (l_out, id_name) for l_in, (l_out, _, id_name) in input_dict.items()},
      cls.arg_VARIABLES: [],
    } for a in Perceptron.allowedActivations], [prob for _ in Perceptron.allowedActivations]

  @classmethod
  def min_transform(cls, nts: TypeShape):
    if nts.dtype not in cls.allowedTypes:
      return None
    s = Shape((DimNames.UNITS, 1))
    result = TypeShape(nts.dtype, s)
    idx = [i for i, d in enumerate(nts.shape.dim) if d.name == DimNames.BATCH]
    if len(idx) > 0:
      idx = idx[0]
      b = nts.shape.dim[idx].size
      if idx + 1 < len(nts.shape.dim) // 2:
        s.dim.insert(0, Shape.Dim(DimNames.BATCH, b))
      else:
        s.dim.append((Shape.Dim(DimNames.BATCH, b)))
    return result

  @classmethod
  def max_transform(cls, nts: TypeShape):
    return cls.min_transform(nts)

  def __init__(self, **kwargs):
    super(Perceptron, self).__init__(**kwargs)
    if not (isinstance(self.attr[self.arg_OUT_TYPE_SHAPE], dict) and
            all([isinstance(nts, TypeShape) and isinstance(label, str) for label, nts in
                 self.attr[self.arg_OUT_TYPE_SHAPE].items()])):
      raise Exception('Wrong output TypeShapes!')

  @property
  def outputs(self) -> Dict[str, TypeShape]:
    return self.attr[self.arg_OUT_TYPE_SHAPE]

  def flops_per_sample(self):
    return (len(self.input_mapping)  # Mult-Acc weighting inputs
            + 1  # Add bias
            + Activations.flops_weight[self.attr[self.arg_ACTIVATION]]  # Nonlinear activation
            )

  def _cls_setstate(self, _function):
    super(Perceptron, self)._cls_setstate(_function)
    self.attr[self.arg_ACTIVATION] = Activations(self.attr[self.arg_ACTIVATION])

  def parameters(self):
    return len(self.input_mapping)

  def mutate(self, prob):
    if random() < prob:
      result = self.__copy__()
      result.attr[self.arg_ACTIVATION] = choice(self.allowedActivations)
      result._id_name = self.getNewName()
      return result
    else:
      return self.__copy__()

  def add_input(self, other_id, other_outputs):
    result = self.__copy__()
    if (not isinstance(other_id, str) or
        not isinstance(other_outputs, dict) or
        any([not isinstance(key, str) or
             not isinstance(value, TypeShape) for key, value in other_outputs.items()])
    ):
      return result
    if len(self.variables) == 0:
      self_ts = next(iter(result.outputs.values()))
      output = choice([label for label, ts in other_outputs.items()
                       if ts.dtype == self_ts.dtype
                       and all([od.name == sd.name
                                for od, sd in zip(ts.shape.dim, self_ts.shape.dim)
                                ])])
      idx = 1
      while True:
        in_label = result.get_in_label(idx)
        if in_label not in result.input_mapping:
          break
        idx += 1
      result.input_mapping[in_label] = (output, other_id)
    else:
      # TODO: extend variable
      raise NotImplementedError()
    result._id_name = self.getNewName()
    return result

  def inputLabels(self) -> List[str]:
    return list(self.input_mapping.keys())
