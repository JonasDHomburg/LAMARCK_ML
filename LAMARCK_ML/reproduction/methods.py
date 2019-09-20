from LAMARCK_ML.reproduction.ancestry import AncestryEntity


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

  def __init__(self, **kwargs):
    self.p = kwargs.get(self.arg_P, .05)
    self.descendants = kwargs.get(self.arg_DESCENDANTS, 1)

  def reproduce(self, pool):
    new_pool = list()
    log = list()
    for ind in pool:
      for _ in range(self.descendants):
        for new_ind in ind.mutate(self.p):
          log.append(AncestryEntity(self.ID, new_ind.id_name, [ind.id_name]))
          new_pool.append(new_ind)
    return new_pool, log


class Recombination(MethodInterface):
  ID = 'RECOMB'

  class Interface():
    def recombine(self, other):
      raise NotImplementedError()

  arg_DESCENDANTS = 'descendants'

  def __init__(self, **kwargs):
    self.descendants = kwargs.get(self.arg_DESCENDANTS, 2)

  def reproduce(self, pool):
    mating_pairs = [(pool[anc1], pool[anc2])
                    for anc1 in range(len(pool)) for anc2 in range(anc1 + 1, len(pool))]
    new_pool = list()
    log = list()
    for anc1, anc2 in mating_pairs:
      for _ in range(self.descendants):
        for new_ind in anc1.recombine(anc2):
          log.append(AncestryEntity(self.ID, new_ind.id_name, [anc1.id_name, anc2.id_name]))
          new_pool.append(new_ind)
        anc1, anc2 = anc2, anc1
    return new_pool, log
