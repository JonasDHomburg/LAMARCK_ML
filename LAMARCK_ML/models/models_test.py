import unittest
from random import random

from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.metrics.interface import MetricInterface
from LAMARCK_ML.models.initialization import InitializationStrategyInterface
from LAMARCK_ML.models.models import GenerationalModel, ModellUtil, NEADone
from LAMARCK_ML.models.models import NoMetric, NoReplaceMethod, NoReproductionMethod, NoSelectionMethod
from LAMARCK_ML.replacement.interface import ReplacementSchemeInterface
from LAMARCK_ML.reproduction.methods import MethodInterface, AncestryEntity
from LAMARCK_ML.selection.interface import SelectionStrategyInterface
from LAMARCK_ML.utils.dataSaver.interface import DataSaverInterface
from LAMARCK_ML.utils.evaluation.interface import EvaluationHelperInterface


class TestGenerationalModel(unittest.TestCase):
  class EndEvaluateException(Exception):
    pass

  class EndSelectException(Exception):
    pass

  class EndReplaceException(Exception):
    pass

  class EndReproduceException(Exception):
    pass

  class EndReproductionStepException(Exception):
    pass

  class TestSaverExceptions(ModellUtil):
    def end_evaluate(self, func):
      def eEWrapper(model):
        raise TestGenerationalModel.EndEvaluateException()

      return eEWrapper

    def end_select(self, func):
      def eSWrapper(model):
        raise TestGenerationalModel.EndSelectException()
        # func()

      return eSWrapper

    def end_replace(self, func):
      def eRWrapper(model):
        raise TestGenerationalModel.EndReplaceException()
        # func()

      return eRWrapper

    def end_reproduce(self, func):
      def eRWrapper(model):
        raise TestGenerationalModel.EndReproduceException()
        # func()

      return eRWrapper

    def end_reproduction_step(self, func):
      def eRSWrapper(model):
        raise TestGenerationalModel.EndReproductionStepException()

      return eRSWrapper

  class TestSaver(DataSaverInterface):
    def __init__(self, **kwargs):
      super(TestGenerationalModel.TestSaver, self).__init__(**kwargs)
      self._idx = kwargs.get('idx', 0)

    def end_evaluate(self, func):
      def eEWrapper(model):
        func()
        model.end_evaluate_hist.append(self._idx)

      return eEWrapper

    def end_select(self, func):
      def eSWrapper(model):
        model.end_select_hist.append(self._idx)
        func()

      return eSWrapper

    def end_replace(self, func):
      def eRWrapper(model):
        model.end_replace_hist.append(self._idx)
        func()

      return eRWrapper

    def end_reproduce(self, func):
      def eRWrapper(model):
        model.end_reproduce_hist.append(self._idx)
        func()

      return eRWrapper

    def end_reproduction_step(self, func):
      def eRSWrapper(model):
        model.end_reproduction_step_hist.append(self._idx)
        func()

      return eRSWrapper

  class TestMetric(MetricInterface):
    ID = 'TMET'

    def evaluate(self, individual, framework):
      return random()

    pass

  class TestReproduce(MethodInterface):
    ID = 'TestRep'

    def reproduce(self, pool):
      if pool is None:
        raise TypeError()
      anc = [AncestryEntity(method=self.ID,
                            descendant=p.id_name,
                            ancestors=[p.id_name]) for p in pool]
      return pool + pool, anc + anc

  class TestEvaluationhelper(EvaluationHelperInterface):
    def evaluate(self, generation, metrics):
      for individual in generation:
        individual.metrics = dict([(m.ID, m.evaluate(individual=individual, framework=None)) for m in metrics])

  class TestSelect(SelectionStrategyInterface):
    def select(self, pool):
      return pool[:int(len(pool) / 2)]

  class TestReplace(ReplacementSchemeInterface):
    def new_generation(self, prev_gen, descendants):
      if descendants is None or prev_gen is None:
        raise TypeError()
      return descendants[0]

  class TestIndividual(IndividualInterface):
    pass

  class TestInitialization(InitializationStrategyInterface):
    def seed_generation(self, func):
      def wrapper(model):
        model._GENERATION = [TestGenerationalModel.TestIndividual() for i in range(6)]
        func()

      return wrapper

  class TestStopper(ModellUtil):
    max_t = 100

    def end_select(self, func):
      def wrapper(model):
        if model.abstract_timestamp >= self.max_t:
          raise NEADone()
        func()

      return wrapper

  def setUp(self):
    self.model = GenerationalModel()
    self.exceptionSaver = self.TestSaverExceptions()
    self.saver = self.TestSaver(idx=1)
    self.saver2 = self.TestSaver(idx=2)
    self.reproduce = self.TestReproduce()
    self.evalHelper = self.TestEvaluationhelper()
    self.select = self.TestSelect()
    self.replace = self.TestReplace()
    self.init_strat = self.TestInitialization()
    self.stopper = self.TestStopper()

    self.model.end_evaluate_hist = []
    self.model.end_select_hist = []
    self.model.end_replace_hist = []
    self.model.end_reproduce_hist = []
    self.model.end_reproduction_step_hist = []
    self.model.seed_generation_hist = []
    self.model.nea_done_hist = []

  def tearDown(self) -> None:
    del self.replace
    del self.select
    del self.evalHelper
    del self.reproduce
    del self.saver
    del self.saver2
    del self.exceptionSaver
    del self.model

  def test_state(self):
    raw_end_evaluate = self.model._end_evaluate
    raw_end_select = self.model._end_select
    raw_end_reproduce = self.model._end_reproduce
    raw_end_reproduction_step = self.model._end_reproduction_step
    raw_end_replace = self.model._end_replace
    raw_seed_generation = self.model._seed_generation

    self.assertTrue(self.model.add(self.exceptionSaver))
    self.assertNotEqual(self.model._end_evaluate, raw_end_evaluate)
    self.assertNotEqual(self.model._end_select, raw_end_select)
    self.assertNotEqual(self.model._end_reproduce, raw_end_reproduce)
    self.assertNotEqual(self.model._end_reproduction_step, raw_end_reproduction_step)
    self.assertNotEqual(self.model._end_replace, raw_end_replace)

    with self.assertRaises(self.EndEvaluateException):
      self.model._end_evaluate()
    with self.assertRaises(self.EndSelectException):
      self.model._end_select()
    with self.assertRaises(self.EndReproduceException):
      self.model._end_reproduce()
    with self.assertRaises(self.EndReproductionStepException):
      self.model._end_reproduction_step()
    with self.assertRaises(self.EndReplaceException):
      self.model._end_replace()

    self.assertTrue(self.model.remove(self.exceptionSaver))
    self.assertEqual(self.model._end_evaluate, raw_end_evaluate)
    self.assertEqual(self.model._end_select, raw_end_select)
    self.assertEqual(self.model._end_reproduce, raw_end_reproduce)
    self.assertEqual(self.model._end_reproduction_step, raw_end_reproduction_step)
    self.assertEqual(self.model._end_replace, raw_end_replace)

    self.assertIsNone(self.model.generation)

    with self.assertRaises(NoMetric):
      self.model._prepare_with_defaults()
    metric = self.TestMetric()
    self.assertTrue(self.model.add(metric))
    self.assertTrue(isinstance(self.model._metrics[0], self.TestMetric))
    self.assertEqual(self.model._metrics[0], metric)

    self.assertTrue(self.model.remove(metric))
    with self.assertRaises(NoMetric):
      self.model._prepare_with_defaults()
    self.assertTrue(self.model.add(metric))

    with self.assertRaises(AttributeError):
      self.model._select()
    with self.assertRaises(NoSelectionMethod):
      self.model._prepare_with_defaults()
    self.assertTrue(self.model.add(self.select))
    with self.assertRaises(TypeError):
      self.model._select()

    with self.assertRaises(NoReproductionMethod):
      self.model._prepare_with_defaults()
    self.assertTrue(self.model.add(self.reproduce))
    with self.assertRaises(TypeError):
      self.model._reproduce()

    with self.assertRaises(NoReplaceMethod):
      self.model._prepare_with_defaults()
    self.assertTrue(self.model.add(self.replace))
    with self.assertRaises(TypeError):
      self.model._replace()

    self.assertFalse(self.model.add(None))

    self.assertEqual(self.model.generation_idx, -1)

    self.model.reset()

    self.assertTrue(self.model.add(self.saver))
    self.assertNotEqual(self.model._end_evaluate, raw_end_evaluate)
    self.assertNotEqual(self.model._end_select, raw_end_select)
    self.assertNotEqual(self.model._end_reproduce, raw_end_reproduce)
    self.assertNotEqual(self.model._end_reproduction_step, raw_end_reproduction_step)
    self.assertNotEqual(self.model._end_replace, raw_end_replace)

    self.model._end_evaluate()
    self.assertListEqual(self.model.end_evaluate_hist, [1])
    self.model._end_select()
    self.assertListEqual(self.model.end_select_hist, [1])
    self.model._end_reproduce()
    self.assertListEqual(self.model.end_reproduce_hist, [1])
    self.model._end_reproduction_step()
    self.assertListEqual(self.model.end_reproduction_step_hist, [1])
    self.model._end_replace()
    self.assertListEqual(self.model.end_replace_hist, [1])

    self.assertTrue(self.model.add(self.saver2))
    self.model._end_evaluate()
    self.assertListEqual(self.model.end_evaluate_hist, [1, 1, 2])
    self.model._end_select()
    self.assertListEqual(self.model.end_select_hist, [1, 2, 1])
    self.model._end_reproduce()
    self.assertListEqual(self.model.end_reproduce_hist, [1, 2, 1])
    self.model._end_reproduction_step()
    self.assertListEqual(self.model.end_reproduction_step_hist, [1, 2, 1])
    self.model._end_replace()
    self.assertListEqual(self.model.end_replace_hist, [1, 2, 1])

    self.assertTrue(self.model.add(self.init_strat))
    self.assertNotEqual(self.model._seed_generation, raw_seed_generation)

    self.assertTrue(self.model.add(self.evalHelper))
    self.assertIs(self.model._evaluation_helper, self.evalHelper)

    self.assertTrue(self.model.add(self.stopper))

    self.model.reset()
    self.model.run()
