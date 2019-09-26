import math
from random import random, choice
from random import sample
from typing import Tuple, List, Dict, Set

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


class Dense(Function, Mutation.Interface):
  # IOLabel.DENSE_OUT = 'DENSE_OUT'
  IOLabel.DENSE_OUT = 'DATA_OUT'
  # IOLabel.DENSE_IN = 'DENSE_IN'
  IOLabel.DENSE_IN = 'DATA_IN'
  allowedTypes = [DFloat, DDouble, DHalf, DInt8, DInt16, DInt32, DInt64]
  _DF_INPUTS = [IOLabel.DENSE_IN]

  arg_UNITS = 'units'
  arg_OUT_NAMED_TYPE_SHAPES = 'outTypeShape'

  __min_f = .5
  __max_f = 1.5

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
        # if _dim.name == DimNames.WIDTH or \
        #     _dim.name == DimNames.HEIGHT or \
        #     _dim.name == DimNames.CHANNEL or \
        if _dim.name == DimNames.UNITS:
          lower_border = max(math.floor(_dim.size * cls.__min_f), (min(2, target_size)
                                                                   if target_size is not None else 2))
          upper_border = math.ceil(_dim.size * cls.__max_f)
          pool = list(range(lower_border + 1, upper_border))
          border_pool = list({lower_border, upper_border})
          if target_size is None or not (lower_border < target_size < upper_border):
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool), 0), len(pool)))
          else:
            pool.remove(target_size)
            pool = sample(pool, k=min(max(max_possibilities - len(border_pool) - 1, 0), len(pool))) + [target_size]
          pool = pool + border_pool
        elif _dim.name == DimNames.BATCH:  # or \
          # _dim.name == DimNames.TIME:
          pool = [_dim.size]
        else:
          invalid_dim = True
          break
        possible_sizes.append(pool)
        names.append(_dim.name)
      if invalid_dim:
        continue

      for dim_combination in Shape.random_dimension_product(possible_sizes):
        out_nts = TypeShape(nts.dtype, Shape(*zip(names, dim_combination)))
        if is_reachable(out_nts, target_output):
          yield ({},
                 {IOLabel.DENSE_OUT: out_nts},
                 {IOLabel.DENSE_IN: label})

  @classmethod
  def generateParameters(cls,
                         input_dict: Dict[str, Tuple[str, Dict[str, TypeShape], str]],
                         expected_outputs: Dict[str, TypeShape],
                         variable_pool: dict = None) -> \
      Tuple[List[Dict[str, object]], List[float]]:
    if len(input_dict) != 1 or \
        len(expected_outputs) != 1:
      return [], []
    input_nts_id, inputs_outputs, _ = input_dict[IOLabel.DENSE_IN]
    in_nts = inputs_outputs[input_nts_id]
    out_label, out_nts = next(iter(expected_outputs.items()))
    if in_nts.dtype != out_nts.dtype:
      return [], []

    inUnits = in_nts.shape.units

    outUnits = out_nts.shape.units

    possibleKernels = []
    possibleBias = []
    if variable_pool is not None:
      possibleKernels = [v for v in variable_pool.get(cls.__name__ + '|kernel', []) if v.shape == (inUnits, outUnits)]
      possibleBias = [v for v in variable_pool.get(cls.__name__ + '|bias', []) if v.shape == (outUnits,)]
    _dict = {cls.arg_ATTRIBUTES: {cls.arg_UNITS: outUnits,
                                  cls.arg_OUT_NAMED_TYPE_SHAPES: {out_label: out_nts},
                                  },
             cls.arg_INPUT_MAPPING: dict(
               [(l_in, (l_out, id_name)) for l_in, (l_out, _, id_name) in input_dict.items()]),
             }

    amount_ = len(possibleBias) + len(possibleKernels)
    init_ = Constant()
    reg_ = NoRegularisation()
    prob_ = 0
    if amount_ > 0:
      prob_ = 1 / 2 / amount_

    _init = [GlorotUniform()]
    _reg = [NoRegularisation()]
    _amount = len(_init) * len(_reg)
    _prob = 1 / 2 / _amount if amount_ > 0 else 1 / _amount
    return ([{**_dict, **{cls.arg_VARIABLES: [k, Variable(**{Variable.arg_DTYPE: out_nts.dtype,
                                                          Variable.arg_TRAINABLE: True,
                                                          Variable.arg_NAME: cls.__name__ + '|bias',
                                                          Variable.arg_SHAPE: (outUnits,),
                                                          Variable.arg_INITIALIZER: init_,
                                                          Variable.arg_REGULARISATION: reg_
                                                             })]}} for k in possibleKernels] +
            [{**_dict, **{cls.arg_VARIABLES: [b, Variable(**{Variable.arg_DTYPE: out_nts.dtype,
                                                             Variable.arg_TRAINABLE: True,
                                                             Variable.arg_NAME: cls.__name__ + '|kernel',
                                                             Variable.arg_SHAPE: (inUnits, outUnits),
                                                             Variable.arg_INITIALIZER: init_,
                                                             Variable.arg_REGULARISATION: reg_,
                                                             })]}} for b in possibleBias] +
            [{**_dict, **{cls.arg_VARIABLES: [Variable(**{Variable.arg_DTYPE: out_nts.dtype,
                                                          Variable.arg_TRAINABLE: True,
                                                          Variable.arg_NAME: cls.__name__ + '|kernel',
                                                          Variable.arg_SHAPE: (inUnits, outUnits),
                                                          Variable.arg_INITIALIZER: _init_,
                                                          Variable.arg_REGULARISATION: _reg_}),
                                              Variable(**{Variable.arg_DTYPE: out_nts.dtype,
                                                          Variable.arg_TRAINABLE: True,
                                                          Variable.arg_NAME: cls.__name__ + '|bias',
                                                          Variable.arg_SHAPE: (outUnits,),
                                                          Variable.arg_INITIALIZER: _init_,
                                                          Variable.arg_REGULARISATION: _reg_})
                                              ]}} for _init_ in _init for _reg_ in _reg]), \
           [prob_ for _ in range(amount_)] + \
           [_prob for _ in range(_amount)]

  def __init__(self, **kwargs):
    super(Dense, self).__init__(**kwargs)
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

    functions = [resetVariable, keepTraining]
    if variable_pool is not None:
      functions.append(replaceVariable)
    result = Dense.__new__(Dense)
    result.__setstate__(self.get_pb())
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
      result._name = Dense.getNewName()
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
      elif _dim.name == DimNames.UNITS:
        s.dim.append(Shape.Dim(_dim.name, int(math.floor(_dim.size * cls.__min_f))))
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
      elif _dim.name == DimNames.UNITS:
        s.dim.append(Shape.Dim(_dim.name, int(math.ceil(_dim.size * cls.__max_f))))
      else:
        return None
    return result
