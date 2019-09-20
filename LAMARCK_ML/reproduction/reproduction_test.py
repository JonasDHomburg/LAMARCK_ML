import random
import unittest

from LAMARCK_ML.individuals import IndividualInterface
from LAMARCK_ML.reproduction import Mutation, Recombination


class TestReproduction(unittest.TestCase):
  def test_mutation(self):
    random.seed()

    class DummyMutInd(IndividualInterface, Mutation.Interface):
      def mutate(self, prob):
        if random.random() < prob:
          return [DummyMutInd()]
        return [self]

    pool = [DummyMutInd() for _ in range(int(1e1))]
    for desc in range(1, 5):
      for p in [0, 1]:
        mut = Mutation(**{
          Mutation.arg_DESCENDANTS: desc,
          Mutation.arg_P: p,
        })
        new_pool, ancestry = mut.reproduce(pool)
        if not p:
          self.assertTrue(all([desc == sum([copy is orig for copy in new_pool]) for orig in pool]))
        else:
          self.assertTrue(all([orig not in new_pool for orig in pool]))
          self.assertEqual(len(new_pool), desc * len(pool))

        self.assertTrue(all(
          [desc == sum([len(anc.ancestors) == 1 and anc.ancestors[0] == orig.id_name
                        and (anc.descendant != orig.id_name if p else
                             anc.descendant == orig.id_name)
                        for anc in ancestry])
           for orig in pool]
        ))
        self.assertTrue(all([anc.method == mut.ID for anc in ancestry]))
    pass

  def test_recombination(self):
    random.seed()

    class DummyRecInd(IndividualInterface, Recombination.Interface):
      def recombine(self, other):
        return [self]

    pool = [DummyRecInd() for _ in range(int(1e1))]
    pairs = len(pool)
    pairs = pairs * (pairs - 1) / 2
    for desc in range(1, 5):
      rec = Recombination(**{
        Recombination.arg_DESCENDANTS: desc
      })
      new_pool, ancestry = rec.reproduce(pool)
      self.assertEqual(len(new_pool), pairs * desc)
      self.assertTrue(all([anc.method == rec.ID for anc in ancestry]))
      self.assertTrue(all([len(anc.ancestors) == 2 for anc in ancestry]))

    pass
