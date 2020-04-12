import unittest
import os

from LAMARCK_ML.architectures.overParameterizedNN import OverParameterizedNeuralNetwork
from LAMARCK_ML.data_util import DimNames, Shape, \
  DFloat, TypeShape, IOLabel
from LAMARCK_ML.utils.compareClass import CompareClass
from LAMARCK_ML.architectures.functions import *
from joblib import Parallel, delayed


@unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
class TestOverParameterizedNN(unittest.TestCase):
  @unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
  def test_instantiate(self):
    for j in range(10):
      batch = 1
      _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch), (DimNames.UNITS, 20)))

      IOLabel.DS1 = 'DS1'
      IOLabel.DS2 = 'DS2'
      inputs = {IOLabel.DS1: (IOLabel.DATA, _data, 'Dataset0'),
                IOLabel.DS2: (IOLabel.DATA, _data, 'Dataset1')}

      outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
      outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
      outputs = {'out0': TypeShape(DFloat, outShape), 'out1': TypeShape(DFloat, outShape1)}
      functions = [Merge, Dense]

      OPNN0 = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
      })
      pb = OPNN0.get_pb()

      pb_OPNN = object.__new__(OverParameterizedNeuralNetwork)
      pb_OPNN.__setstate__(pb)

      self.assertEqual(OPNN0, pb_OPNN)

      OPNN1 = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
      })
      for i in range(1):
        mutOPNN = OPNN0.recombine(OPNN1)[0]
        for f in mutOPNN.functions:
          self.assertTrue(f.id_name in mutOPNN.meta_functions)
        functions = {f.id_name for f in mutOPNN.functions}
        for out_id, (label, f_id) in mutOPNN.output_mapping.items():
          self.assertTrue(f_id in functions)
        for k in range(10):
          mutOPNN = mutOPNN.mutate(1)[0]
          for f in mutOPNN.functions:
            self.assertTrue(f.id_name in mutOPNN.meta_functions)
            if isinstance(f, Merge):
              self.assertEqual(len(f.inputs), 2)
            functions = {f.id_name for f in mutOPNN.functions}
            for out_id, (label, f_id) in mutOPNN.output_mapping.items():
              self.assertTrue(f_id in functions)
    pass

  @unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
  def test_recombination(self):
    batch = 1
    _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch), (DimNames.UNITS, 20)))

    IOLabel.DS1 = 'DS1'
    IOLabel.DS2 = 'DS2'
    inputs = {IOLabel.DS1: (IOLabel.DATA, _data, 'Dataset0'),
              IOLabel.DS2: (IOLabel.DATA, _data, 'Dataset1')}

    outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
    outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
    outputs = {'out0': TypeShape(DFloat, outShape), 'out1': TypeShape(DFloat, outShape1)}
    functions = [Merge, Dense]
    OPNN0 = OverParameterizedNeuralNetwork(**{
      OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
      OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
      OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
      OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
    })
    pb = OPNN0.get_pb()

    pb_OPNN = object.__new__(OverParameterizedNeuralNetwork)
    pb_OPNN.__setstate__(pb)

    self.assertEqual(OPNN0, pb_OPNN)

    OPNN1 = OverParameterizedNeuralNetwork(**{
      OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
      OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
      OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
      OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
    })

    mut1 = OPNN0.recombine(OPNN1)[0]
    mut2 = OPNN1.recombine(OPNN0)[0]
    self.assertEqual(len(mut1.meta_functions), len(mut2.meta_functions))
    mut3 = mut1.recombine(mut2)[0]
    mut4 = mut2.recombine(mut1)[0]
    self.assertEqual(len(mut3.meta_functions), len(mut4.meta_functions))
    self.assertEqual(len(mut3.meta_functions), len(mut1.meta_functions))
    self.assertEqual(len(mut4.meta_functions), len(mut1.meta_functions))
    pass

  def test_mut_cross_update(self):

    def _test_():
      IOLabel.DS1 = 'DS1'
      IOLabel.DS2 = 'DS2'

      batch = 1
      _data = TypeShape(DFloat, Shape((DimNames.BATCH, batch), (DimNames.UNITS, 20)))
      inputs = {IOLabel.DS1: (IOLabel.DATA, _data, 'Dataset0'),
                IOLabel.DS2: (IOLabel.DATA, _data, 'Dataset1')}

      outShape = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 10))
      outShape1 = Shape((DimNames.BATCH, batch), (DimNames.UNITS, 15))
      outputs = {'out0': TypeShape(DFloat, outShape), 'out1': TypeShape(DFloat, outShape1)}
      functions = [Merge, Dense]

      prim = 'a'
      sec0 = 'b'
      sec1 = 'c'
      cmp = CompareClass(**{
        CompareClass.arg_PRIMARY_ALPHA: 1,
        CompareClass.arg_PRIMARY_OBJECTIVE: prim,
        CompareClass.arg_PRIMARY_THRESHOLD: 0.5,
        CompareClass.arg_SECONDARY_OBJECTIVES: {sec0: -10 / 0.1,
                                                sec1: -20 / 0.1}
      })
      one = {prim: 0.7,
             sec0: 20,
             sec1: 40}
      other = {prim: 0.6,
               sec0: 25,
               sec1: 30}

      OPNN0 = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
        OverParameterizedNeuralNetwork.arg_CONSCIOUSNESS: 1,
      })
      OPNN0.cmp = cmp
      OPNN0.update_state(**{OverParameterizedNeuralNetwork.meta_QUALITY: one})

      OPNN1 = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
        OverParameterizedNeuralNetwork.arg_CONSCIOUSNESS: 1,
      })
      OPNN1.cmp = cmp
      OPNN1.update_state(**{OverParameterizedNeuralNetwork.meta_QUALITY: other})

      OPNN2 = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
        OverParameterizedNeuralNetwork.arg_CONSCIOUSNESS: 1,
      })
      OPNN2.cmp = cmp
      OPNN2.update_state(**{OverParameterizedNeuralNetwork.meta_QUALITY: other})

      OPNN3 = OverParameterizedNeuralNetwork(**{
        OverParameterizedNeuralNetwork.arg_INPUTS: inputs,
        OverParameterizedNeuralNetwork.arg_OUTPUT_TARGETS: outputs,
        OverParameterizedNeuralNetwork.arg_FUNCTIONS: functions,
        OverParameterizedNeuralNetwork.arg_MAX_BRANCH: 2,
        OverParameterizedNeuralNetwork.arg_CONSCIOUSNESS: 1,
      })
      OPNN3.cmp = cmp
      OPNN3.update_state(**{OverParameterizedNeuralNetwork.meta_QUALITY: other})

      OPNN4_ = OPNN0.recombine(OPNN1)[0]
      OPNN4 = OPNN4_.mutate(.2)[0]
      OPNN4.cmp = cmp
      prev = set(OPNN4.meta_function_consciousness.keys())
      OPNN4.update_state(**{OverParameterizedNeuralNetwork.meta_QUALITY: {
        prim: 0.65,
        sec0: 30,
        sec1: 35,
      }})
      after = set(OPNN4.meta_function_consciousness.keys())
      diff = prev.difference(after)

      for f, meta in OPNN4.meta_functions.items():
        self.assertFalse(f in diff)

      for (d0, ts0), label, (d1, ts1) in OPNN4.meta_edges:
        if isinstance(ts0, TypeShape):
          self.assertFalse(ts1 in diff)
        else:
          self.assertFalse(ts0 in diff)

      OPNN5_ = OPNN2.recombine(OPNN3)[0]
      OPNN5 = OPNN5_.mutate(.2)[0]
      OPNN5.cmp = cmp
      prev = set(OPNN5.meta_function_consciousness.keys())
      OPNN5.update_state(**{OverParameterizedNeuralNetwork.meta_QUALITY: {
        prim: 0.71,
        sec0: 27,
        sec1: 32,
      }})
      after = set(OPNN5.meta_function_consciousness.keys())
      diff = prev.difference(after)

      for f, meta in OPNN5.meta_functions.items():
        self.assertFalse(f in diff)

      for (d0, ts0), label, (d1, ts1) in OPNN5.meta_edges:
        if isinstance(ts0, TypeShape):
          self.assertFalse(ts1 in diff)
        else:
          self.assertFalse(ts0 in diff)

      OPNN6_ = OPNN4.recombine(OPNN5)[0]
      OPNN6 = OPNN6_.mutate(.2)[0]

    a = Parallel(n_jobs=6)(delayed(_test_)() for _ in range(10000))

  pass
