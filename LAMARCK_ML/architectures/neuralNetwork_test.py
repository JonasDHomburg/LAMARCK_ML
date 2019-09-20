import unittest
import os

from LAMARCK_ML.architectures.functions import *
from LAMARCK_ML.architectures.neuralNetwork import NeuralNetwork
from LAMARCK_ML.data_util import DimNames, Shape, \
  DFloat, TypeShape, IOLabel
import networkx as nx
import time


@unittest.skipIf((os.environ.get('test_fast', False) in {'True','true', '1'}), 'time consuming')
class TestNeuralNetwork(unittest.TestCase):
  def test_instantiation_USD_outputTypeShapes(self):
    batch = 3
    _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch),
                                    (DimNames.CHANNEL, 3), (DimNames.HEIGHT, 4),
                                    (DimNames.WIDTH, 5)))

    outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 60))
    self.assertRaises(InvalidFunctionType, NeuralNetwork, **{
      NeuralNetwork.arg_INPUTS: {'data_in', (_data, 'Dataset')},
      NeuralNetwork.arg_OUTPUT_TARGETS: {'out0': TypeShape(DFloat, outShape)}})

  def test_instantiation_USD_ONTS_Dense_Merge(self):
    for i in range(100):
      batch = 1
      _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch), (DimNames.UNITS, 20)))

      IOLabel.DS1 = 'DS1'
      IOLabel.DS2 = 'DS2'
      inputs = {IOLabel.DS1: (IOLabel.DATA, _data, 'Dataset'),
                IOLabel.DS2: (IOLabel.DATA, _data, 'Dataset')}

      outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
      outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
      outputs = {'out0': TypeShape(DFloat, outShape), 'out1': TypeShape(DFloat, outShape1)}
      functions = [Merge, Dense]
      NN = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: inputs,
                            NeuralNetwork.arg_OUTPUT_TARGETS: outputs,
                            NeuralNetwork.arg_FUNCTIONS: functions,
                            NeuralNetwork.arg_MAX_BRANCH: 2})
      self.assertIsNotNone(NN)

      pb = NN.get_pb()
      state = NN.__getstate__()

      NN_pb = NeuralNetwork.__new__(NeuralNetwork)
      NN_pb.__setstate__(pb)
      self.assertIsNot(NN, NN_pb)

      NN_state = NeuralNetwork.__new__(NeuralNetwork)
      NN_state.__setstate__(state)
      self.assertIsNot(NN, NN_state)

      NN_mut = NN.mutate(100)[0]
      self.assertEqual(pb, NN.get_pb())
      self.assertIsNot(NN, NN_mut)
      self.assertNotEqual(NN, NN_mut)

      f_ids = dict([(_id, None) for _, _id in NN_mut.inputs.values()])
      for _f in NN_mut.functions:
        f_ids[_f.id_name] = _f

      for _f in NN_mut.functions:
        for _f_input, (other_output, other_id) in _f.inputs.items():
          if other_id not in f_ids:
            self.assertTrue(False)

      stack = [f_id for _, f_id in NN_mut.output_mapping.values()]
      required_ids = set()
      while stack:
        f_id = stack.pop()
        required_ids.add(f_id)
        f_ = f_ids.get(f_id)
        if f_ is not None:
          stack.extend([f_id for _, f_id in f_.inputs.values()])
      self.assertSetEqual(required_ids, set(f_ids.keys()))

      NN_mut = NN.mutate(100)[0]
      self.assertEqual(pb, NN.get_pb())
      self.assertIsNot(NN, NN_mut)
      self.assertNotEqual(NN, NN_mut)

      f_ids = dict([(_id, None) for _, _id in NN_mut.inputs.values()])
      for _f in NN_mut.functions:
        f_ids[_f.id_name] = _f

      for _f in NN_mut.functions:
        for _f_input, (other_output, other_id) in _f.inputs.items():
          if other_id not in f_ids:
            self.assertTrue(False)

      stack = [f_id for _, f_id in NN_mut.output_mapping.values()]
      required_ids = set()
      while stack:
        f_id = stack.pop()
        required_ids.add(f_id)
        f_ = f_ids.get(f_id)
        if f_ is not None:
          stack.extend([f_id for _, f_id in f_.inputs.values()])
      if len(required_ids) > len(f_ids):
        print('Unused Functions!', required_ids.difference(f_ids.keys()), set(f_ids.keys()).difference(required_ids))
      self.assertSetEqual(required_ids, set(f_ids.keys()))

      NN_mut = NN.mutate(0)[0]
      NN_mut._id_name = NN._id_name
      self.assertNotEqual(NN, NN_mut)

  def test_instantiation_Conv2D_Pool2D_Flatten(self):
    for i in range(10):
      batch = 1
      _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch),
                                      (DimNames.HEIGHT, 64),
                                      (DimNames.WIDTH, 64),
                                      (DimNames.CHANNEL, 3)))
      _target = TypeShape(DFloat, Shape((DimNames.BATCH, batch),
                                        (DimNames.UNITS, 100),
                                        ))
      outputs = {'out0': _target}
      IOLabel.DS = 'DS'
      inputs = {IOLabel.DS: (IOLabel.DATA, _data, 'Dataset')}
      functions = [Conv2D, Pooling2D, Flatten]
      NN1 = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: inputs,
                             NeuralNetwork.arg_OUTPUT_TARGETS: outputs,
                             NeuralNetwork.arg_FUNCTIONS: functions})
      self.assertIsNotNone(NN1)
      # print(i)
      pb = NN1.get_pb()
      state = NN1.__getstate__()

      NN_pb = NeuralNetwork.__new__(NeuralNetwork)
      NN_pb.__setstate__(pb)
      self.assertIsNot(NN1, NN_pb)

      NN_state = NeuralNetwork.__new__(NeuralNetwork)
      NN_state.__setstate__(state)
      self.assertIsNot(NN1, NN_state)

      NN_mut = NN1.mutate(100)
      self.assertIsNot(NN1, NN_mut)
      self.assertNotEqual(NN1, NN_mut)
      NN_mut = NN1.mutate(0)
      self.assertIsNot(NN1, NN_mut)
      self.assertNotEqual(NN1, NN_mut)

      NN2 = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: inputs,
                             NeuralNetwork.arg_OUTPUT_TARGETS: outputs,
                             NeuralNetwork.arg_FUNCTIONS: functions})

      NN_rec = NN1.recombine(NN2)[0]
      self.assertIsNotNone(NN_rec)
      f_ids = dict([(_id, None) for _, _id in NN_rec.inputs.values()])
      for _f in NN_rec.functions:
        f_ids[_f.id_name] = _f

      for _f in NN_rec.functions:
        for _f_input, (other_output, other_id) in _f.inputs.items():
          if other_id not in f_ids:
            self.assertTrue(False)

      stack = [f_id for _, f_id in NN_rec.output_mapping.values()]
      required_ids = set()
      while stack:
        f_id = stack.pop()
        required_ids.add(f_id)
        f_ = f_ids.get(f_id)
        if f_ is not None:
          stack.extend([f_id for _, f_id in f_.inputs.values()])
      self.assertSetEqual(required_ids, set(f_ids.keys()))

  def test_instantiation_Conv2D_Pool2D_Flatten_Dense(self):
    for i in range(10):
      batch = 1
      _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch),
                                      (DimNames.HEIGHT, 32),
                                      (DimNames.WIDTH, 32),
                                      (DimNames.CHANNEL, 3)))
      _target = TypeShape(DFloat, Shape((DimNames.BATCH, batch),
                                        (DimNames.UNITS, 10),
                                        ))
      outputs = {'out0': _target}
      IOLabel.DS = 'DS'
      inputs = {IOLabel.DS: (IOLabel.DATA, _data, 'Dataset')}
      functions = [Conv2D, Pooling2D, Flatten, Dense]
      NN = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: inputs,
                            NeuralNetwork.arg_OUTPUT_TARGETS: outputs,
                            NeuralNetwork.arg_FUNCTIONS: functions})
      self.assertIsNotNone(NN)
      pb = NN.get_pb()
      state = NN.__getstate__()

      NN_pb = NeuralNetwork.__new__(NeuralNetwork)
      NN_pb.__setstate__(pb)
      self.assertIsNot(NN, NN_pb)

      NN_state = NeuralNetwork.__new__(NeuralNetwork)
      NN_state.__setstate__(state)
      self.assertIsNot(NN, NN_state)

      NN_mut = NN.mutate(100)
      self.assertIsNot(NN, NN_mut)
      self.assertNotEqual(NN, NN_mut)
      NN_mut = NN.mutate(0)
      self.assertIsNot(NN, NN_mut)
      self.assertNotEqual(NN, NN_mut)

  def test_recombination_Dense_Merge(self):
    for i in range(100):
      batch = 1
      _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch), (DimNames.UNITS, 20)))

      IOLabel.DS1 = 'DS1'
      IOLabel.DS2 = 'DS2'
      inputs = {IOLabel.DS1: (IOLabel.DATA, _data, 'Dataset'),
                IOLabel.DS2: (IOLabel.DATA, _data, 'Dataset')}

      outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
      outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
      outputs = {'out0': TypeShape(DFloat, outShape),
                 'out1': TypeShape(DFloat, outShape1)}
      functions = [Merge, Dense]
      NN1 = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: inputs,
                             NeuralNetwork.arg_OUTPUT_TARGETS: outputs,
                             NeuralNetwork.arg_FUNCTIONS: functions,
                             NeuralNetwork.arg_RECOMBINATION_PROBABILITY: 1.0})
      self.assertIsNotNone(NN1)
      NN2 = NeuralNetwork(**{NeuralNetwork.arg_INPUTS: inputs,
                             NeuralNetwork.arg_OUTPUT_TARGETS: outputs,
                             NeuralNetwork.arg_FUNCTIONS: functions,
                             NeuralNetwork.arg_RECOMBINATION_PROBABILITY: 1.0})
      self.assertIsNotNone(NN2)
      NN_rec = NN1.recombine(NN2)[0]

      f_ids = dict([(_id, None) for _, _id in NN_rec.inputs.values()])
      for _f in NN_rec.functions:
        f_ids[_f.id_name] = _f

      for _f in NN_rec.functions:
        for _f_input, (other_output, other_id) in _f.inputs.items():
          if other_id not in f_ids:
            self.assertTrue(False)

      stack = [f_id for _, f_id in NN_rec.output_mapping.values()]
      required_ids = set()
      while stack:
        f_id = stack.pop()
        required_ids.add(f_id)
        f_ = f_ids.get(f_id)
        if f_ is not None:
          stack.extend([f_id for _, f_id in f_.inputs.values()])
      self.assertSetEqual(required_ids, set(f_ids.keys()))
      # print(i)

  @unittest.skip('debug')
  def test_reachable(self):
    # target = TypeShape(DFloat, Shape((DimNames.BATCH, 1),
    #                                  (DimNames.UNITS, 20)))
    input_shape = TypeShape(DFloat, Shape((DimNames.BATCH, 1),
                                          (DimNames.HEIGHT, 32),
                                          (DimNames.WIDTH, 32),
                                          (DimNames.CHANNEL, 3)
                                          ))
    depth = 8
    for i in range(1, 100):
      # input_shape = TypeShape(DFloat, Shape((DimNames.BATCH, 1), (DimNames.UNITS, i)))
      target = TypeShape(DFloat, Shape((DimNames.BATCH, 1), (DimNames.UNITS, i * 10)))
      print()
      print(input_shape)
      print(target)
      print(NeuralNetwork.reachable(input_nts=input_shape,
                                    target_nts=target,
                                    max_depth=depth,
                                    function_pool={Conv2D, Flatten}))
      print(list(Dense.possible_output_shapes(input_ntss={IOLabel.DEFAULT: input_shape},
                                              target_output=target,
                                              is_reachable=
                                              lambda x, y: NeuralNetwork.reachable(x, y, depth - 1, {Dense, Merge}),
                                              )
                 ))
    pass

  @unittest.skip('debugging')
  def test_simple_path(self):
    ntss = {IOLabel.DEFAULT: TypeShape(DFloat, Shape((DimNames.BATCH, 1),
                                                     (DimNames.UNITS, 23)))}
    target = TypeShape(DFloat, Shape((DimNames.BATCH, 1),
                                     (DimNames.UNITS, 154)))
    depth = 5
    debug_node = 'debug'

    before = time.time()
    for _ in range(1):
      NeuralNetwork.reachable(next(iter(ntss)), target, depth, {Dense, Merge})
    print('Time', time.time() - before)

    print(NeuralNetwork.reachable(next(iter(ntss)), target, depth, {Dense, Merge}))
    print(ntss)
    runs = 10000
    fails = 0
    for i in range(runs):
      blueprint = nx.DiGraph()
      blueprint.add_node(debug_node,
                         ntss=ntss,
                         DataFlowObj=None)
      out_node, nts_id, nodes = next(NeuralNetwork.simple_path(input_node=debug_node,
                                                               input_ntss=ntss,
                                                               output_shape=target,
                                                               output_label=IOLabel.DEFAULT,
                                                               blueprint=blueprint,
                                                               min_depth=0,
                                                               max_depth=depth,
                                                               function_pool={Dense, Merge},
                                                               ), (None, None, None))
      if out_node is None:
        # print(i, 'Error')
        fails += 1
      # else:
      #   print(i, 'Success')
    print('percentage failed:', fails / runs)
    pass

  @unittest.skip('debugging')
  def test_func_children(self):
    ntss = {IOLabel.DEFAULT: TypeShape(DFloat, Shape((DimNames.BATCH, 1),
                                                     (DimNames.HEIGHT, 10),
                                                     (DimNames.WIDTH, 10),
                                                     (DimNames.CHANNEL, 2)))}
    target = TypeShape(DFloat, Shape((DimNames.BATCH, 1),
                                     (DimNames.UNITS, 200)
                                     # (DimNames.HEIGHT, 32),
                                     # (DimNames.WIDTH, 32),
                                     # (DimNames.CHANNEL, 3)
                                     ))

    _f = Flatten
    for _, out_nts, _ in _f.possible_output_shapes(
        ntss, target, lambda x, y: NeuralNetwork.reachable(x, y, 0, {Flatten}), 10):
      print(next(iter(out_nts.values())))

    pass
