import math
import time
from enum import Enum
from random import sample
from typing import Tuple, List, Dict, Set
from random import choice, random

from LAMARCK_ML.architectures.functions.interface import Function
from LAMARCK_ML.architectures.variables import Variable
from LAMARCK_ML.architectures.variables.initializer import *
from LAMARCK_ML.architectures.variables.regularisation import *
from LAMARCK_ML.data_util import DimNames, TypeShape, IOLabel
from LAMARCK_ML.data_util.dataType import \
  DHalf, \
  DFloat, \
  DDouble, \
  DInt64, \
  DInt32, \
  DInt16, \
  DInt8
from LAMARCK_ML.data_util.shape import Shape
from LAMARCK_ML.reproduction.methods import Mutation


class Conv2D(Function, Mutation.Interface):
  # IOLabel.CONV2D_IN = 'CONV2D_IN'
  IOLabel.CONV2D_IN = 'DATA_IN'
  # IOLabel.CONV2D_OUT = 'CONV2D_OUT'
  IOLabel.CONV2D_OUT = 'DATA_OUT'

  class Padding(Enum):
    SAME = 'SAME'
    VALID = 'VALID'

  allowedTypes = [DFloat, DDouble, DHalf, DInt8, DInt16, DInt32, DInt64]
  _DF_INPUTS = [IOLabel.CONV2D_IN]

  arg_OUT_NAMED_TYPE_SHAPES = 'outTypeShape'
  arg_KERNEL_WIDTH = 'kernel_width'
  arg_KERNEL_HEIGHT = 'kernel_height'
  arg_STRIDE_WIDTH = 'stride_width'
  arg_STRIDE_HEIGHT = 'stride_height'
  arg_PADDING = 'padding'
  arg_FILTER = 'channel'
  arg_IN_WIDTH = 'in_width'
  arg_IN_HEIGHT = 'in_height'
  arg_IN_CHANNEL = 'in_channel'

  __min_f_hw = .5
  __max_f_hw = 1
  __min_f_c = .5
  __max_f_c = 1.5

  @classmethod
  def possible_output_shapes(cls,
                             input_ntss: Dict[str, TypeShape],
                             target_output: TypeShape,
                             is_reachable,
                             max_possibilities: int = 10) -> \
      List[Tuple[Dict[str, TypeShape], Dict[str, TypeShape], Dict[str, str]]]:

    target_shape = target_output.shape

    for label, nts in input_ntss.items():
      _prev = time.time()
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
          border_pool = list({lower_border, upper_border})
          if target_size is None or not (lower_border < target_size < upper_border):
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool), 0), len(pool)))
          else:
            pool.remove(target_size)
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool) - 1, 0), len(pool))) + [target_size]
          pool = pool + border_pool
        elif _dim.name == DimNames.CHANNEL:
          lower_border = max(math.floor(_dim.size * cls.__min_f_c), (min(2, target_size)
                                                                     if target_size is not None else 2))
          upper_border = math.ceil(_dim.size * cls.__max_f_c)
          pool = list(range(lower_border + 1, upper_border))
          border_pool = list({lower_border, upper_border})
          if target_size is None or not (lower_border < target_size < upper_border):
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool), 0), len(pool)))
          else:
            pool.remove(target_size)
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool) - 1, 0), len(pool))) + [target_size]
          pool = pool + border_pool
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
          yield ({},
                 {IOLabel.CONV2D_OUT: out_nts},
                 {IOLabel.CONV2D_IN: label})

  @classmethod
  def configurations(cls, h_i, h_o, w_i, w_o, c):
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

    def filter_range(in_, out_):
      lower_limit = 1
      upper_limit = in_ - out_ + 1
      return list(range(lower_limit, upper_limit + 1))

    configurations = list()

    for s_h in stride_range(h_i, h_o):
      for s_w in stride_range(w_i, w_o):
        for k_h in range(1, 10):
          for k_w in range(1, 10):
            configurations.append({
              cls.arg_KERNEL_HEIGHT: k_h,
              cls.arg_KERNEL_WIDTH: k_w,
              cls.arg_STRIDE_HEIGHT: s_h,
              cls.arg_STRIDE_WIDTH: s_w,
              cls.arg_PADDING: Conv2D.Padding.SAME.value,
              cls.arg_FILTER: c
            })
    for k_w in filter_range(w_i, w_o):
      for k_h in filter_range(h_i, h_o):
        for s_h in stride_range(h_i - k_h + 1, h_o):
          for s_w in stride_range(w_i - k_w + 1, w_o):
            configurations.append({
              cls.arg_KERNEL_HEIGHT: k_h,
              cls.arg_KERNEL_WIDTH: k_w,
              cls.arg_STRIDE_HEIGHT: s_h,
              cls.arg_STRIDE_WIDTH: s_w,
              cls.arg_PADDING: Conv2D.Padding.VALID.value,
              cls.arg_FILTER: c
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

    input_nts_id, input_outputs, input_id = input_dict[IOLabel.CONV2D_IN]
    in_nts = input_outputs[input_nts_id]
    out_label, out_nts = next(iter(expected_outputs.items()))
    if in_nts.dtype != out_nts.dtype:
      return [], []

    allowed_dimensions = [DimNames.BATCH, DimNames.CHANNEL, DimNames.WIDTH, DimNames.HEIGHT]
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
    c_i = in_nts.shape[DimNames.CHANNEL]
    configurations = cls.configurations(
      h_i=h_i,
      h_o=h_o,
      w_i=w_i,
      w_o=w_o,
      c=out_nts.shape[DimNames.CHANNEL])
    result_with_var = list()
    result_without_var = list()
    for config in configurations:
      result_without_var.append({cls.arg_ATTRIBUTES: {**config,
                                                      **{cls.arg_OUT_NAMED_TYPE_SHAPES: {out_label: out_nts},
                                                         cls.arg_IN_WIDTH: w_i,
                                                         cls.arg_IN_HEIGHT: h_i,
                                                         cls.arg_IN_CHANNEL: c_i}},
                                 cls.arg_INPUT_MAPPING: dict(
                                   [(l_in, (l_out, id_name)) for l_in, (l_out, _, id_name) in input_dict.items()]),
                                 cls.arg_VARIABLES: [
                                   Variable(**{
                                     Variable.arg_DTYPE: out_nts.dtype,
                                     Variable.arg_TRAINABLE: True,
                                     Variable.arg_NAME: cls.__name__ + '|kernel',
                                     Variable.arg_SHAPE: (
                                       config.get(cls.arg_KERNEL_HEIGHT),
                                       config.get(cls.arg_KERNEL_WIDTH),
                                       in_nts.shape[DimNames.CHANNEL],
                                       out_nts.shape[DimNames.CHANNEL]),
                                     Variable.arg_INITIALIZER: GlorotUniform(),
                                     Variable.arg_REGULARISATION: NoRegularisation()
                                   }),
                                   Variable(**{
                                     Variable.arg_DTYPE: out_nts.dtype,
                                     Variable.arg_TRAINABLE: True,
                                     Variable.arg_NAME: cls.__name__ + '|bias',
                                     Variable.arg_SHAPE: (
                                       out_nts.shape[DimNames.CHANNEL],),
                                     Variable.arg_INITIALIZER: GlorotUniform(),
                                     Variable.arg_REGULARISATION: NoRegularisation()
                                   })
                                 ]})
      possibleKernels = [v for v in variable_pool.get(cls.__name__ + '|kernel', [])
                         if v.shape == (config.get(cls.arg_KERNEL_HEIGHT),
                                        config.get(cls.arg_KERNEL_WIDTH),
                                        c_i,
                                        config.get(cls.arg_FILTER))]
      for kernel in possibleKernels:
        result_with_var.append({cls.arg_ATTRIBUTES: {**config,
                                                     **{cls.arg_OUT_NAMED_TYPE_SHAPES: {out_label: out_nts},
                                                        cls.arg_IN_WIDTH: w_i,
                                                        cls.arg_IN_HEIGHT: h_i,
                                                        cls.arg_IN_CHANNEL: c_i}},
                                cls.arg_INPUT_MAPPING: dict(
                                  [(l_in, (l_out, id_name)) for l_in, (l_out, _, id_name) in input_dict.items()]),
                                cls.arg_VARIABLES: [kernel,
                                                    Variable(**{
                                                      Variable.arg_DTYPE: out_nts.dtype,
                                                      Variable.arg_TRAINABLE: True,
                                                      Variable.arg_NAME: cls.__name__ + '|bias',
                                                      Variable.arg_SHAPE: (
                                                        out_nts.shape[DimNames.CHANNEL]),
                                                      Variable.arg_INITIALIZER: GlorotUniform(),
                                                      Variable.arg_REGULARISATION: NoRegularisation()
                                                    })
                                                    ]})
      possibleBias = [v for v in variable_pool.get(cls.__name__ + '|bias', [])
                      if v.shape == (out_nts.shape[DimNames.CHANNEL],)]
      for bias in possibleBias:
        result_with_var.append({cls.arg_ATTRIBUTES: {**config,
                                                     **{cls.arg_OUT_NAMED_TYPE_SHAPES: {out_label: out_nts},
                                                        cls.arg_IN_WIDTH: w_i,
                                                        cls.arg_IN_HEIGHT: h_i,
                                                        cls.arg_IN_CHANNEL: c_i}},
                                cls.arg_INPUT_MAPPING: dict(
                                  [(l_in, (l_out, id_name)) for l_in, (l_out, _, id_name) in input_dict.items()]),
                                cls.arg_VARIABLES: [
                                  Variable(**{
                                    Variable.arg_DTYPE: out_nts.dtype,
                                    Variable.arg_TRAINABLE: True,
                                    Variable.arg_NAME: cls.__name__ + '|kernel',
                                    Variable.arg_SHAPE: (
                                      config.get(cls.arg_KERNEL_HEIGHT),
                                      config.get(cls.arg_KERNEL_WIDTH),
                                      c_i,
                                      out_nts.shape[DimNames.CHANNEL]),
                                    Variable.arg_INITIALIZER: GlorotUniform(),
                                    Variable.arg_REGULARISATION: NoRegularisation()
                                  }),
                                  bias
                                ]})
    result_params = list()
    result_prob = list()
    amount_var = len(result_with_var)
    if amount_var > 0:
      prob = 1 / 2 / amount_var
      result_params.extend(result_with_var)
      result_prob.extend([prob for _ in range(amount_var)])
    amount_without_var = len(result_without_var)
    prob = 1 / amount_without_var
    if amount_var > 0:
      prob /= 2
    result_params.extend(result_without_var)
    result_prob.extend([prob for _ in range(amount_without_var)])
    return result_params, result_prob

  def __init__(self, **kwargs):
    super(Conv2D, self).__init__(**kwargs)
    if not (isinstance(self.attr[self.arg_OUT_NAMED_TYPE_SHAPES], dict) and
            all([isinstance(nts, TypeShape) and isinstance(label, str) for label, nts in
                 self.attr[self.arg_OUT_NAMED_TYPE_SHAPES].items()])):
      raise Exception('Wrong output TypeShapes!')

  @property
  def outputs(self) -> Set[TypeShape]:
    return self.attr[self.arg_OUT_NAMED_TYPE_SHAPES]

  def mutate(self, prob, variable_pool=None):
    def resetVariable(v):
      v.value = None
      v.trainable = True
      return v

    def keepTraining(v):
      v.trainable = True
      return v

    def replaceVariable(v):
      variable = choice(
        [_v for _v in variable_pool.get(self.__name__ + '|kernel', []) if v.shape == _v.shape])
      return variable

    result = Conv2D.__new__(Conv2D)
    result.__setstate__(self.get_pb())

    if random() < .8:
      functions = [resetVariable, keepTraining]
      if variable_pool is not None:
        functions.append(replaceVariable)
      new_variables = list()
      changed = False
      for _v in result.variables:
        if random() < prob:
          new_variable = choice(functions)(_v)
          changed = True
        else:
          new_variable = _v
        new_variables.append(new_variable)
      result.variables = new_variables
      if changed:
        result._name = Conv2D.getNewName()
    else:
      if random() < prob:
        result._name = Conv2D.getNewName()
        out_nts = self.attr[self.arg_OUT_NAMED_TYPE_SHAPES][IOLabel.CONV2D_OUT]
        h_o = out_nts.shape[DimNames.HEIGHT]
        w_o = out_nts.shape[DimNames.WIDTH]
        c = out_nts.shape[DimNames.CHANNEL]
        config = choice(Conv2D.configurations(h_i=self.attr[self.arg_IN_HEIGHT],
                                              h_o=h_o,
                                              w_i=self.attr[self.arg_IN_WIDTH],
                                              w_o=w_o,
                                              c=c))
        result.attr = {**result.attr, **config}
        result.variables = [Variable(**{Variable.arg_DTYPE: out_nts.dtype,
                                        Variable.arg_TRAINABLE: True,
                                        Variable.arg_NAME: self.__class__.__name__ + '|kernel',
                                        Variable.arg_SHAPE: (
                                          config.get(self.arg_KERNEL_HEIGHT),
                                          config.get(self.arg_KERNEL_WIDTH),
                                          self.attr[self.arg_IN_CHANNEL],
                                          out_nts.shape[DimNames.CHANNEL]),
                                        Variable.arg_INITIALIZER: GlorotUniform(),
                                        Variable.arg_REGULARISATION: NoRegularisation()
                                        }),
                            Variable(**{
                              Variable.arg_DTYPE: out_nts.dtype,
                              Variable.arg_TRAINABLE: True,
                              Variable.arg_NAME: self.__class__.__name__ + '|bias',
                              Variable.arg_SHAPE: (
                                out_nts.shape[DimNames.HEIGHT],
                                out_nts.shape[DimNames.WIDTH],
                                out_nts.shape[DimNames.CHANNEL]),
                              Variable.arg_INITIALIZER: GlorotUniform(),
                              Variable.arg_REGULARISATION: NoRegularisation()
                            })]
    return result

  @classmethod
  def min_transform(cls, nts):
    if nts.dtype not in cls.allowedTypes:
      return None
    s = Shape()
    result = TypeShape(nts.dtype, s)
    for _dim in nts.shape.dim:
      if _dim.name == DimNames.BATCH:
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif _dim.name == DimNames.CHANNEL:
        s.dim.append(Shape.Dim(_dim.name, int(math.floor(_dim.size * cls.__min_f_c))))
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
      if _dim.name == DimNames.BATCH:
        s.dim.append(Shape.Dim(_dim.name, _dim.size))
      elif _dim.name == DimNames.CHANNEL:
        s.dim.append(Shape.Dim(_dim.name, int(math.ceil(_dim.size * cls.__max_f_c))))
      elif _dim.name == DimNames.WIDTH or \
          _dim.name == DimNames.HEIGHT:
        s.dim.append(Shape.Dim(_dim.name, int(math.ceil(_dim.size * cls.__max_f_hw))))
      else:
        return None
    return result
