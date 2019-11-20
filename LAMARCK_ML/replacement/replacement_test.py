import random
import unittest
import os

from LAMARCK_ML.individuals import IndividualInterface
from LAMARCK_ML.replacement import \
  GenerationalReplacement, \
  NElitism, \
  NWeakElitism, \
  DeleteN, \
  DeleteNLast
from LAMARCK_ML.reproduction.methods import Mutation


@unittest.skipIf((os.environ.get('test_fast', False) in {'True', 'true', '1'}), 'time consuming')
class TestReplacementSchemes(unittest.TestCase):
  class OneMetricIndividual(IndividualInterface, Mutation.Interface):
    def __init__(self):
      self._fitness = random.random()
      self._random_loc = random.random() * 1e2
      self._id_name = self.getNewName()
      self.metrics = dict()
      self.attr = dict()

    def __sub__(self, other):
      return abs(self._random_loc - other._random_loc)

    def mutate(self, prob):
      return self

  @staticmethod
  def reverse_cmp(x, y):
    return 0 if x == y else 1 if x < y else -1

  def setUp(self) -> None:
    self.prev_gen = [TestReplacementSchemes.OneMetricIndividual() for _ in range(int(1e2))]
    self.descendants = [[TestReplacementSchemes.OneMetricIndividual() for _ in range(int(1e2))] for _ in range(10)]

  def tearDown(self) -> None:
    del self.prev_gen
    del self.descendants

  def test_GenerationalReplacement(self):
    rep_obj = GenerationalReplacement()
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(self.descendants[-1]), len(new_gen))
    self.assertListEqual(new_gen, self.descendants[-1])
    pass

  def test_NElitism(self):
    n = random.randint(1, 10)
    rep_obj = NElitism(**{
      NElitism.arg_N: n
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(self.descendants[-1]) + n, len(new_gen))
    self.assertEqual(len([ind for ind in new_gen if ind in self.prev_gen]), n)
    rep_obj = NElitism(**{
      NElitism.arg_N: n,
      NElitism.arg_CMP: TestReplacementSchemes.reverse_cmp
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(self.descendants[-1]) + n, len(new_gen))
    self.assertEqual(len([ind for ind in new_gen if ind in self.prev_gen]), n)
    pass

  def test_NWeakElitism(self):
    n = random.randint(1, 10)
    rep_obj = NWeakElitism(**{
      NWeakElitism.arg_N: n
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(new_gen), len(self.prev_gen) + n)
    self.assertEqual(len([ind for ind in new_gen if ind not in self.descendants[-1]]), n)
    self.assertEqual(len([ind for ind in new_gen if ind in self.descendants[-1]]), len(self.descendants[-1]))
    rep_obj = NWeakElitism(**{
      NWeakElitism.arg_N: n,
      NWeakElitism.arg_CMP: TestReplacementSchemes.reverse_cmp
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(new_gen), len(self.prev_gen) + n)
    pass

  def test_DeleteN(self):
    n = random.randint(1, 10)
    rep_obj = DeleteN(**{
      DeleteN.arg_N: n
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(new_gen), len(self.prev_gen))
    self.assertEqual(len([ind for ind in new_gen if ind in self.prev_gen]) + n, len(self.prev_gen))
    pass

  def test_DeleteNLast(self):
    n = random.randint(1, 10)
    rep_obj = DeleteN(**{
      DeleteNLast.arg_N: n
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(new_gen), len(self.prev_gen))
    self.assertEqual(len([ind for ind in new_gen if ind in self.prev_gen]) + n, len(self.prev_gen))
    rep_obj = DeleteN(**{
      DeleteNLast.arg_N: n,
      DeleteNLast.arg_CMP: TestReplacementSchemes.reverse_cmp
    })
    new_gen = rep_obj.new_generation(self.prev_gen, self.descendants)
    self.assertEqual(len(new_gen), len(self.prev_gen))
    self.assertEqual(len([ind for ind in new_gen if ind in self.prev_gen]) + n, len(self.prev_gen))
    pass
