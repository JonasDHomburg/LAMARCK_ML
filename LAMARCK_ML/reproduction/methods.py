from LAMARCK_ML.reproduction.ancestry import AncestryEntity

from random import sample
from joblib import Parallel, delayed

import os

try:
  cpu_avail = int(os.environ['CPU_AVAIL'])
except:
  cpu_avail = 1


class MethodInterface():
  ID = 'INVALID'

  def reproduce(self, pool):
    raise NotImplementedError()

  pass


class Mutation(MethodInterface):
  ID = 'MUTATE'

  class Interface():
    def mutate(self, prob):
      raise NotImplementedError()

  arg_P = 'p'
  arg_DESCENDANTS = 'descendants'
  arg_LIMIT = 'limit'

  def __init__(self, **kwargs):
    self.p = kwargs.get(self.arg_P, .05)
    self.descendants = kwargs.get(self.arg_DESCENDANTS, 1)
    self.limit = kwargs.get(self.arg_LIMIT)

  def reproduce(self, pool):
    def mutate(ind, p):
      return ind.mutate(p), ind

    new_pool = list()
    log = list()
    iter_list = [ind for ind in pool for _ in range(self.descendants)]

    for new_inds, ind in Parallel(n_jobs=cpu_avail, require='sharedmem')(
        delayed(mutate)(_ind, self.p) for _ind in iter_list):
      for new_ind in new_inds:
        log.append(AncestryEntity(self.ID, new_ind.id_name, [ind.id_name]))
        new_pool.append(new_ind)
    if self.limit is not None:
      comb = list(zip(new_pool, log))
      new_pool, log = zip(*sample(comb, k=min(len(comb), self.limit)))
    return new_pool, log


class Recombination(MethodInterface):
  ID = 'RECOMB'

  class Interface():
    def recombine(self, other):
      raise NotImplementedError()

  arg_DESCENDANTS = 'descendants'
  arg_LIMIT = 'limit'

  def __init__(self, **kwargs):
    self.descendants = kwargs.get(self.arg_DESCENDANTS, 2)
    self.limit = kwargs.get(self.arg_LIMIT)

  def reproduce(self, pool):
    def recomb(anc1, anc2):
      return anc1.recombine(anc2), anc1, anc2

    mating_pairs = [(pool[anc1], pool[anc2])
                    for anc1 in range(len(pool)) for anc2 in range(anc1 + 1, len(pool))]
    new_pool = list()
    log = list()
    iter_list = list()
    for anc1, anc2 in mating_pairs:
      for _ in range(self.descendants):
        iter_list.append((anc1, anc2))
        anc1, anc2 = anc2, anc1
    for recomb_desc, anc1, anc2 in Parallel(n_jobs=cpu_avail, require='sharedmem')(
        delayed(recomb)(anc1, anc2) for anc1, anc2 in iter_list):
      for new_ind in recomb_desc:
        log.append(AncestryEntity(self.ID, new_ind.id_name, [anc1.id_name, anc2.id_name]))
        new_pool.append(new_ind)
    if self.limit is not None:
      comb = list(zip(new_pool, log))
      new_pool, log = zip(*sample(comb, k=min(len(comb), self.limit)))
    return new_pool, log
