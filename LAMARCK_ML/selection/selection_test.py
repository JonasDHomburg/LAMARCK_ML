import random
import unittest
import os

from LAMARCK_ML.individuals import IndividualInterface
from LAMARCK_ML.selection import \
  LinearRankingSelection, \
  ExponentialRankingSelection, \
  MaxDiversitySelection, \
  ProportionalSelection, \
  TournamentSelection, \
  TruncationSelection, \
  GreedyOverSelection


@unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
class TestSelectionStrategies(unittest.TestCase):
  class OneMetricIndividual(IndividualInterface):
    def __init__(self):
      self._fitness = random.random()
      self._random_loc = random.random() * 1e2

    def __sub__(self, other):
      return abs(self._random_loc - other._random_loc)

    def __eq__(self, other):
      return self is other

    pass

  @staticmethod
  def reverse_cmp(x, y):
    return -1 if x > y else 1 if y > x else 0

  def setUp(self) -> None:
    self.OMpopulation = [TestSelectionStrategies.OneMetricIndividual() for _ in range(int(1e4))]
    self.select = int(1e2)

  def tearDown(self) -> None:
    del self.OMpopulation

  def test_LinearRankingSelection(self):
    sel_obj = LinearRankingSelection(**{
      LinearRankingSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    sel_obj = LinearRankingSelection(**{
      LinearRankingSelection.arg_CMP: TestSelectionStrategies.reverse_cmp,
      LinearRankingSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  def test_ExponentialRankingSelection(self):
    sel_obj = ExponentialRankingSelection(**{
      ExponentialRankingSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    sel_obj = ExponentialRankingSelection(**{
      ExponentialRankingSelection.arg_CMP: TestSelectionStrategies.reverse_cmp,
      ExponentialRankingSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  def test_MaxDiversitySelection(self):
    sel_obj = MaxDiversitySelection(**{
      MaxDiversitySelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  def test_ProportionalSelection(self):
    sel_obj = ProportionalSelection(**{
      ProportionalSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  def test_TournamentSelection(self):
    sel_obj = TournamentSelection(**{
      TournamentSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    sel_obj = TournamentSelection(**{
      TournamentSelection.arg_CMP: TestSelectionStrategies.reverse_cmp,
      TournamentSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  def test_TruncationSelection(self):
    sel_obj = TruncationSelection(**{
      TruncationSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    sel_obj = TruncationSelection(**{
      TruncationSelection.arg_CMP: TestSelectionStrategies.reverse_cmp,
      TruncationSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  def test_GreedyOverSelection(self):
    sel_obj = GreedyOverSelection(**{
      GreedyOverSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    sel_obj = GreedyOverSelection(**{
      GreedyOverSelection.arg_CMP: TestSelectionStrategies.reverse_cmp,
      GreedyOverSelection.arg_LIMIT: self.select
    })
    selected = sel_obj.select(self.OMpopulation)
    self.assertEqual(len(selected), self.select)
    pass

  pass
