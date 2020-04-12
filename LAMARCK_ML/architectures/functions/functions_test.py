import random
import time
import unittest
import os
from typing import Dict

import numpy as np

from LAMARCK_ML.architectures import DataFlow
from LAMARCK_ML.architectures.functions import *
from LAMARCK_ML.architectures.variables import Variable
from LAMARCK_ML.data_util import DimNames as DN, IOLabel, Shape, BaseType, DDouble, TypeShape
from LAMARCK_ML.architectures.neuralNetwork import NeuralNetwork


@unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
class TestFunction(unittest.TestCase):
  def allTypes(self, _type=BaseType):
    for cls in _type.__subclasses__(_type):
      yield cls
      for r in self.allTypes(_type=cls):
        yield r

  def randomVariable(self):
    return Variable(dtype=DDouble, value=np.random.random((3, 7, 4, 9)), trainable=True)

  def test_getClassByName(self):
    self.assertRaises(InvalidFunctionClass, Function.getClassByName, 'Function')

  def test_getNewName(self):
    Function.resetNames()
    classes = [Function]
    names = []
    for c in classes:
      for i in range(random.randint(1, 10)):
        obj = c(variables=[self.randomVariable() for _ in range(random.randint(1, 3))], input_mapping={}, dtype=DDouble)
        names.append(obj._id_name)
    keys = [k for k in Function.usedNames.keys()]
    keys.sort()
    names.sort()
    self.assertListEqual(keys, names)
    Function.resetNames()
    self.assertEqual(len(Function.usedNames.keys()), 0)

  class dummySubClass(Function):
    pass

  def test_pb(self):
    dummy_in = TestFunction.dummySubClass(variables=[self.randomVariable() for _ in range(random.randint(1, 3))],
                                          input_mapping={},
                                          attributes={'int_val': random.randint(0, 100),
                                                      'float_val': random.random(),
                                                      'string_val': "asdfasdf",
                                                      'bool_val': True,
                                                      },
                                          dtype=DDouble)
    IOLabel.OTHER = 'OTHER'
    obj = TestFunction.dummySubClass(variables=[self.randomVariable() for _ in range(random.randint(1, 3))],
                                     input_mapping={IOLabel.DEFAULT: (IOLabel.DEFAULT, dummy_in),
                                                    IOLabel.OTHER: (IOLabel.DEFAULT, dummy_in.id_name)},
                                     attributes={'int_val': random.randint(0, 100),
                                                 'float_val': random.random(),
                                                 'string_val': "asdfasdf",
                                                 'bool_val': True,
                                                 'bool_val2': False,
                                                 'bytes_val': random.randint(0, 100).to_bytes(10, byteorder='big'),
                                                 'shape_val': Shape((DN.BATCH, -1),
                                                                    (DN.WIDTH, 16),
                                                                    (DN.HEIGHT, 16),
                                                                    (DN.WIDTH, 3)),
                                                 'dtype_val': DDouble,
                                                 'list_val': [1, 2, 3, 4],
                                                 },
                                     dtype=DDouble)
    pb = obj.get_pb()
    self.assertIsNotNone(pb)
    state = obj.__getstate__()
    self.assertIsNotNone(state)
    new_obj = Function.get_instance(state)
    self.assertIs(obj, new_obj)

    new_obj = Function.__new__(Function)
    new_obj.__setstate__(state)
    self.assertEqual(obj, new_obj)
    self.assertIsNot(obj, new_obj)
    new_obj = Function.__new__(Function)
    new_obj.__setstate__(pb)
    self.assertEqual(obj, new_obj)
    self.assertIsNot(obj, new_obj)
    new_obj2 = Function.get_instance(state)
    self.assertIs(new_obj, new_obj2)

    pb.attr[0].v.int_val = -1
    new_obj = Function.__new__(Function)
    new_obj.__setstate__(pb)
    self.assertNotEqual(obj, new_obj)

  def test_subclassException(self):
    cls = TestFunction.dummySubClass
    self.assertRaises(NotImplementedError, cls.generateParameters, *(None, None))

  def test_generateParameters(self):
    self.assertRaises(NotImplementedError, Function.generateParameters,
                      # **{'input_dict': {'test': Shape()}, 'outputShape': Shape()})
                      **{'input_dict': {'test': None}, 'expected_outputs': set(), 'variable_pool': dict()})

  def test_possible_output_shapes(self):
    self.assertRaises(NotImplementedError, Function.possible_output_shapes,
                      **{'input_ntss': None, 'target_output': None, 'is_reachable': None})

  @unittest.skip('Debugging')
  def test_sampling(self):
    samples = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]
    ]

    option_count = list()
    counter = 1
    for s in samples:
      option_count.append(counter)
      counter *= len(s)
    option_count = option_count[::-1]
    print(option_count)

    probabilities_dict = dict()
    index_lists = [len(s) for s in samples]

    probabilities_dict[tuple()] = [option_count[0] for _ in samples[0]]

    samples_left = True
    sample_depth = len(samples)
    while samples_left:
      current_sample = []
      for depth in range(sample_depth):
        pool = index_lists[depth]
        c = probabilities_dict.get(tuple(current_sample))
        if c is None:
          c = [option_count[depth] for _ in samples[depth]]
        p = np.asarray(c)
        print(p)
        p = p / np.sum(p)
        s = np.random.choice(pool, size=1, replace=False, p=p)[0]

        c[s] -= 1
        probabilities_dict[tuple(current_sample)] = c
        current_sample.append(samples[depth][s])
        if depth == 0:
          samples_left = sum(c) > 0
      print(current_sample)
      time.sleep(1)

    pass


@unittest.skipIf((os.environ.get('test_fast', False) in {'True','true', '1'}), 'time consuming')
class TestSubFunctions(unittest.TestCase):
  class DummyDF(DataFlow):
    _outputs = None
    _id_name = 'DummyDF'

    @property
    def outputs(self) -> Dict[str, TypeShape]:
      return self._outputs

    @property
    def id_name(self) -> str:
      return self._id_name

  def test_Dense(self):
    IOLabel.DUMMY = 'DUMMY'
    IOLabel.DUMMY2 = 'DUMMY2'
    _shape = Shape((DN.BATCH, -1), (DN.UNITS, 16))
    _input = {IOLabel.DUMMY: TypeShape(DDouble, _shape)}
    _output = TypeShape(DDouble, _shape)
    _expected_output = TypeShape(DDouble, _shape)
    pos = list(Dense.possible_output_shapes(
      input_ntss=_input,
      target_output=_output,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Dense})
    ))
    self.assertTrue(any([_expected_output in out.values() for _, out, _ in pos]))
    self.assertTrue(all([len(remaining) == 0 for remaining, _, _ in pos]))
    self.assertTrue(
      all([IOLabel.DENSE_IN in mapping and mapping[IOLabel.DENSE_IN] == IOLabel.DUMMY for _, _, mapping in pos]))

    dummyDF = TestSubFunctions.DummyDF()
    dummyDF._outputs = {IOLabel.DUMMY: TypeShape(DDouble, _shape)}
    for _, out, _ in pos:
      for parameters in Dense.generateParameters(
          input_dict={IOLabel.DENSE_IN: (IOLabel.DUMMY, dummyDF.outputs, dummyDF.id_name)},
          expected_outputs={IOLabel.DUMMY2: _output},
          variable_pool={},
      )[0]:
        # check if parameters are correct?
        _dense = Dense(**parameters)
        pb = _dense.get_pb()
        self.assertIsNotNone(pb)
        state = _dense.__getstate__()
        self.assertIsNotNone(state)

        new_dense = Dense.__new__(Dense)
        new_dense.__setstate__(pb)
        self.assertEqual(_dense, new_dense)
        self.assertIsNot(_dense, new_dense)

        new_dense = Dense.__new__(Dense)
        new_dense.__setstate__(state)
        self.assertEqual(_dense, new_dense)
        self.assertIsNot(_dense, new_dense)

        m_dense = _dense.mutate(100)
        self.assertNotEqual(_dense, m_dense)
        m_dense = _dense.mutate(0)
        self.assertEqual(_dense, m_dense)
    pass

  def test_Merge(self):
    IOLabel.DUMMY = 'DUMMY'
    IOLabel.DUMMY2 = 'DUMMY2'
    _input = {IOLabel.DUMMY: TypeShape(DDouble, Shape((DN.BATCH, -1), (DN.UNITS, 16)))}
    _output = TypeShape(DDouble, Shape((DN.BATCH, -1), (DN.UNITS, 24)))
    pos = list(Merge.possible_output_shapes(
      input_ntss=_input,
      target_output=_output,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Merge}),
    ))
    self.assertTrue(any([_output in out.values() for _, out, _ in pos]))
    self.assertTrue(all(
      [len(remaining) == 1 and all([nts_label == IOLabel.MERGE_OTHER for nts_label in remaining.keys()])
       for remaining, _, _ in pos]))
    self.assertTrue(
      all([IOLabel.MERGE_IN in mapping and mapping[IOLabel.MERGE_IN] == IOLabel.DUMMY for _, _, mapping in pos]))
    dummyDF = TestSubFunctions.DummyDF()
    dummyDF._outputs = {IOLabel.DUMMY: TypeShape(DDouble, Shape((DN.BATCH, -1), (DN.UNITS, 16)))}
    for remaining, out, _ in pos:
      dummyDF2 = TestSubFunctions.DummyDF()
      nts = next(iter(remaining.values()))
      dummyDF2._outputs = {IOLabel.DUMMY2: TypeShape(nts.dtype, nts.shape)}
      for parameters in Merge.generateParameters(
          input_dict={IOLabel.MERGE_IN: (IOLabel.DUMMY, dummyDF.outputs, dummyDF.id_name),
                      IOLabel.MERGE_OTHER: (IOLabel.DUMMY2, dummyDF2.outputs, dummyDF2.id_name)},
          expected_outputs={_output},
          variable_pool={},
      )[0]:
        _merge = Merge(**parameters)
        pb = _merge.get_pb()
        self.assertIsNotNone(pb)
        state = _merge.__getstate__()
        self.assertIsNotNone(state)

        new_merge = Merge.__new__(Merge)
        new_merge.__setstate__(pb)
        self.assertEqual(_merge, new_merge)
        self.assertIsNot(_merge, new_merge)

        new_merge = Merge.__new__(Merge)
        new_merge.__setstate__(state)
        self.assertEqual(_merge, new_merge)
        self.assertIsNot(_merge, new_merge)
    pass

  def test_Conv2D(self):
    IOLabel.DUMMY1 = 'DUMMY1'
    IOLabel.DUMMY2 = 'DUMMY2'
    _shape = Shape((DN.BATCH, -1),
                   (DN.WIDTH, 64),
                   (DN.HEIGHT, 64),
                   (DN.CHANNEL, 4))
    shape_ = Shape((DN.BATCH, -1),
                   (DN.WIDTH, 32),
                   (DN.HEIGHT, 32),
                   (DN.CHANNEL, 6))
    _input = {IOLabel.DUMMY1: TypeShape(DDouble, _shape)}
    _output = TypeShape(DDouble, shape_)
    _expected_output = TypeShape(DDouble, shape_)
    pos = list(Conv2D.possible_output_shapes(
      input_ntss=_input,
      target_output=_output,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Conv2D})
    ))
    self.assertTrue(any([_expected_output in out.values() for _, out, _ in pos]))
    self.assertTrue(all([len(remaining) == 0 for remaining, _, _ in pos]))
    self.assertTrue(
      all([IOLabel.CONV2D_IN in mapping and mapping[IOLabel.CONV2D_IN] == IOLabel.DUMMY1 for _, _, mapping in pos]))

    dummyDF = TestSubFunctions.DummyDF()
    dummyDF._outputs = {IOLabel.DUMMY1: TypeShape(DDouble, _shape)}
    for _, out, _ in pos:
      for parameters in Conv2D.generateParameters(
          input_dict={IOLabel.CONV2D_IN: (IOLabel.DUMMY1, dummyDF.outputs, dummyDF.id_name)},
          expected_outputs={IOLabel.CONV2D_OUT: _output},
          variable_pool={},
      )[0]:
        _conv2D = Conv2D(**parameters)
        pb = _conv2D.get_pb()
        self.assertIsNotNone(pb)
        state = _conv2D.__getstate__()
        self.assertIsNotNone(state)

        new_conv2D = Conv2D.__new__(Conv2D)
        new_conv2D.__setstate__(pb)
        self.assertEqual(_conv2D, new_conv2D)
        self.assertIsNot(_conv2D, new_conv2D)

        new_conv2D = Conv2D.__new__(Conv2D)
        new_conv2D.__setstate__(state)
        self.assertEqual(_conv2D, new_conv2D)
        self.assertIsNot(_conv2D, new_conv2D)

        m_conv2D = _conv2D.mutate(100)
        self.assertNotEqual(_conv2D, m_conv2D)
        m_conv2D = _conv2D.mutate(0)
        self.assertEqual(_conv2D, m_conv2D)
        self.assertIsNot(_conv2D, m_conv2D)
    pass

  def test_Pool2D(self):
    IOLabel.DUMMY1 = 'DUMMY1'
    IOLabel.DUMMY2 = 'DUMMY2'
    _shape = Shape((DN.BATCH, -1),
                   (DN.WIDTH, 64),
                   (DN.HEIGHT, 64),
                   (DN.CHANNEL, 4))
    shape_ = Shape((DN.BATCH, -1),
                   (DN.WIDTH, 32),
                   (DN.HEIGHT, 32),
                   (DN.CHANNEL, 4))
    _input = {IOLabel.DUMMY1: TypeShape(DDouble, _shape)}
    _output = TypeShape(DDouble, shape_)
    _expected_output = TypeShape(DDouble, shape_)
    pos = list(Pooling2D.possible_output_shapes(
      input_ntss=_input,
      target_output=_output,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Pooling2D})
    ))
    self.assertTrue(any([_expected_output in out.values() for _, out, _ in pos]))
    self.assertTrue(all([len(remaining) == 0 for remaining, _, _ in pos]))
    self.assertTrue(
      all(
        [IOLabel.POOLING2D_IN in mapping and mapping[IOLabel.POOLING2D_IN] == IOLabel.DUMMY1 for _, _, mapping in pos]))

    dummyDF = TestSubFunctions.DummyDF()
    dummyDF._outputs = {IOLabel.DUMMY1: TypeShape(DDouble, _shape)}
    for _, out, _ in pos:
      for parameters in Pooling2D.generateParameters(
          input_dict={IOLabel.POOLING2D_IN: (IOLabel.DUMMY1, dummyDF.outputs, dummyDF.id_name)},
          expected_outputs={IOLabel.POOLING2D_OUT: _output},
          variable_pool={},
      )[0]:
        _pool2D = Pooling2D(**parameters)
        pb = _pool2D.get_pb()
        self.assertIsNotNone(pb)
        state = _pool2D.__getstate__()
        self.assertIsNotNone(state)

        new_pool2D = Pooling2D.__new__(Pooling2D)
        new_pool2D.__setstate__(pb)
        self.assertEqual(_pool2D, new_pool2D)
        self.assertIsNot(_pool2D, new_pool2D)

        new_pool2D = Pooling2D.__new__(Pooling2D)
        new_pool2D.__setstate__(state)
        self.assertEqual(_pool2D, new_pool2D)
        self.assertIsNot(_pool2D, new_pool2D)

        m_pool2D = _pool2D.mutate(100)
        self.assertNotEqual(_pool2D, m_pool2D)
        m_pool2D = _pool2D.mutate(0)
        self.assertEqual(_pool2D, m_pool2D)
        self.assertIsNot(_pool2D, m_pool2D)

  def test_Flatten(self):
    IOLabel.DUMMY1 = 'DUMMY1'
    IOLabel.DUMMY2 = 'DUMMY2'
    _shape = Shape((DN.BATCH, -1),
                   (DN.WIDTH, 8),
                   (DN.HEIGHT, 8),
                   (DN.CHANNEL, 32))
    shape_ = Shape((DN.BATCH, -1),
                   (DN.UNITS, 2048))
    _input = {IOLabel.DUMMY1: TypeShape(DDouble, _shape)}
    _output = TypeShape(DDouble, shape_)
    _expected_output = TypeShape(DDouble, shape_)
    pos = list(Flatten.possible_output_shapes(
      input_ntss=_input,
      target_output=_output,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Flatten})
    ))
    self.assertTrue(any([_expected_output in out.values() for _, out, _ in pos]))
    self.assertTrue(all([len(remaining) == 0 for remaining, _, _ in pos]))
    self.assertTrue(
      all([IOLabel.FLATTEN_IN in mapping and mapping[IOLabel.FLATTEN_IN] == IOLabel.DUMMY1 for _, _, mapping in pos]))

    dummyDF = TestSubFunctions.DummyDF()
    dummyDF._outputs = {IOLabel.DUMMY1: TypeShape(DDouble, _shape)}
    for _, out, _ in pos:
      for parameters in Flatten.generateParameters(
          input_dict={IOLabel.FLATTEN_IN: (IOLabel.DUMMY1, dummyDF.outputs, dummyDF.id_name)},
          expected_outputs={IOLabel.DUMMY2: _output},
          variable_pool={},
      )[0]:
        _flatten = Flatten(**parameters)
        pb = _flatten.get_pb()
        self.assertIsNotNone(pb)
        state = _flatten.__getstate__()
        self.assertIsNotNone(state)

        new_flatten = Flatten.__new__(Flatten)
        new_flatten.__setstate__(pb)
        self.assertEqual(_flatten, new_flatten)
        self.assertIsNot(_flatten, new_flatten)

        new_flatten = Flatten.__new__(Flatten)
        new_flatten.__setstate__(state)
        self.assertEqual(_flatten, new_flatten)
        self.assertIsNot(_flatten, new_flatten)

  def test_Softmax(self):
    IOLabel.DUMMY1 = 'DUMMY1'
    IOLabel.DUMMY2 = 'DUMMY2'
    shape0 = Shape((DN.BATCH, -1),
                   (DN.WIDTH, 8),
                   (DN.HEIGHT, 8),
                   (DN.CHANNEL, 32))
    shape1 = Shape((DN.BATCH, -1),
                   (DN.UNITS, 10))
    output0 = TypeShape(DDouble, shape0)
    output1 = TypeShape(DDouble, shape1)
    input0 = {IOLabel.DUMMY1: output0}
    input1 = {IOLabel.DUMMY1: output1}
    pos0 = list(Softmax.possible_output_shapes(
      input_ntss=input0,
      target_output=output0,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Softmax})
    ))
    self.assertTrue(any([output0 in out.values() for _, out, _ in pos0]))
    self.assertTrue(all([len(remaining) == 0 for remaining, _, _ in pos0]))
    self.assertTrue(all([IOLabel.SOFTMAX_IN in mapping and mapping[IOLabel.SOFTMAX_IN] == IOLabel.DUMMY1
                         for _, _, mapping in pos0]))
    pos1 = list(Softmax.possible_output_shapes(
      input_ntss=input1,
      target_output=output1,
      is_reachable=lambda x, y: NeuralNetwork.reachable(x, y, 1, {Softmax})
    ))
    self.assertTrue(any([output1 in out.values() for _, out, _ in pos1]))
    self.assertTrue(all([len(remaining) == 0 for remaining, _, _ in pos1]))
    self.assertTrue(all([IOLabel.SOFTMAX_IN in mapping and mapping[IOLabel.SOFTMAX_IN] == IOLabel.DUMMY1
                         for _, _, mapping in pos1]))

    dummyDF = TestSubFunctions.DummyDF()
    dummyDF._outputs = {IOLabel.DUMMY1: TypeShape(DDouble, shape0)}
    for _, out, _ in pos0:
      for parameters in Softmax.generateParameters(
          input_dict={IOLabel.SOFTMAX_IN: (IOLabel.DUMMY1, dummyDF.outputs, dummyDF.id_name)},
          expected_outputs={IOLabel.DUMMY2: output0},
          variable_pool={},
      )[0]:
        _softmax = Softmax(**parameters)
        pb = _softmax.get_pb()
        self.assertIsNotNone(pb)
        state = _softmax.__getstate__()
        self.assertIsNotNone(state)

        new_softmax = Softmax.__new__(Softmax)
        new_softmax.__setstate__(pb)
        self.assertEqual(_softmax, new_softmax)
        self.assertIsNot(_softmax, new_softmax)

        new_softmax = Softmax.__new__(Softmax)
        new_softmax.__setstate__(state)
        self.assertEqual(_softmax, new_softmax)
        self.assertIsNot(_softmax, new_softmax)

    for _, out, _ in pos1:
      for parameters in Softmax.generateParameters(
          input_dict={IOLabel.SOFTMAX_IN: (IOLabel.DUMMY1, dummyDF.outputs, dummyDF.id_name)},
          expected_outputs={IOLabel.DUMMY2: output1},
          variable_pool={}
      )[0]:
        _softmax = Softmax(**parameters)
        pb = _softmax.get_pb()
        self.assertIsNotNone(pb)
        state = _softmax.__getstate__()
        self.assertIsNotNone(state)

        new_softmax = Softmax.__new__(Softmax)
        new_softmax.__setstate__(pb)
        self.assertEqual(_softmax, new_softmax)
        self.assertIsNot(_softmax, new_softmax)

        new_softmax = Softmax.__new__(Softmax)
        new_softmax.__setstate__(state)
        self.assertEqual(_softmax, new_softmax)
        self.assertIsNot(_softmax, new_softmax)
