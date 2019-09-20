import os
import unittest
from random import random

from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.metrics.interface import MetricInterface
from LAMARCK_ML.models.initialization import InitializationStrategyInterface
from LAMARCK_ML.models.models import GenerationalModel, ModellUtil, NEADone
from LAMARCK_ML.replacement.interface import ReplacementSchemeInterface
from LAMARCK_ML.reproduction import AncestryEntity
from LAMARCK_ML.reproduction.methods import MethodInterface
from LAMARCK_ML.selection.interface import SelectionStrategyInterface
from LAMARCK_ML.utils import ModelStateSaverLoader
from LAMARCK_ML.utils.evaluation.interface import EvaluationHelperInterface


class TestModelStateSaverLoader(unittest.TestCase):
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
        # individual.set_up_eval()
        individual.metrics = dict([(m.ID, m.evaluate(self=m, individual=individual, framework=None)) for m in metrics])
        # individual.tear_down_eval()

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
        model._GENERATION = [TestModelStateSaverLoader.TestIndividual() for i in range(6)]
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
    self.reproduce = self.TestReproduce()
    self.evalHelper = self.TestEvaluationhelper()
    self.select = self.TestSelect()
    self.replace = self.TestReplace()
    self.init_strat = self.TestInitialization()
    self.stopper = self.TestStopper()
    self.state_file = './debug.pb'
    self.stateSaver = ModelStateSaverLoader(**{
      ModelStateSaverLoader.arg_FILE: self.state_file,
      ModelStateSaverLoader.arg_EVALUATION: True,
      ModelStateSaverLoader.arg_REPLACEMENT: True,
    })

  def tearDown(self) -> None:
    del self.replace
    del self.select
    del self.evalHelper
    del self.reproduce
    del self.model
    del self.stateSaver

  def test_state(self):
    self.model.add([self.TestMetric,
                    self.select,
                    self.reproduce,
                    self.replace,
                    self.evalHelper,
                    self.init_strat,
                    self.stopper,
                    self.stateSaver])
    self.model.reset()
    self.model.run()
    self.assertEqual(self.model.generation_idx, 100)

    self.stopper.max_t = 200

    self.model.reset()
    self.model._prepare_with_defaults()
    self.assertEqual(self.model.generation_idx, 100)
    self.model.run()
    self.assertEqual(self.model.generation_idx, 200)

    os.remove(self.state_file)
