import numpy as np

from LAMARCK_ML.individuals import sortingClass
from LAMARCK_ML.replacement.interface import ReplacementSchemeInterface
from LAMARCK_ML.reproduction.methods import Mutation


class SortingBasedReplacement(ReplacementSchemeInterface):
  arg_CMP = 'cmp'

  def __init__(self, **kwargs):
    super(SortingBasedReplacement, self).__init__(**kwargs)
    self.cmp = kwargs.get(self.arg_CMP)


class GenerationalReplacement(ReplacementSchemeInterface):
  """
  Generational Replacement
  """

  def __init__(self, **kwargs):
    super(GenerationalReplacement, self).__init__(**kwargs)

  def __str__(self):
    return 'GR'

  def new_generation(self, prev_gen, descendants):
    return list(descendants[-1])


class NElitism(SortingBasedReplacement):
  """
  N Elitism
  """
  arg_N = 'n'

  def __init__(self, **kwargs):
    super(NElitism, self).__init__()
    self.__n = kwargs.get(self.arg_N, 1)

  def __str__(self):
    return "NE n=%i" % self.__n

  def new_generation(self, prev_gen, descendants):
    return descendants[-1] + \
           [sc.obj for sc in sorted([sortingClass(obj=p, cmp=self.cmp) for p in prev_gen], reverse=True)][:self.__n]


class NWeakElitism(SortingBasedReplacement):
  """
  N Weak Elitism
  """
  arg_N = 'n'
  arg_P = 'p'

  def __init__(self, **kwargs):
    """
    :param n: Integers - Number of elite individuals
    :param p: Float - Mutation probability
    """
    super(NWeakElitism, self).__init__(**kwargs)
    self.__n = kwargs.get(self.arg_N, 1)
    self.__p = kwargs.get(self.arg_P, .1)

  def __str__(self):
    return 'NWE n=%i p=%01.2f' % (self.__n, self.__p)

  def new_generation(self, prev_gen, descendants):
    if not (all([isinstance(ind, Mutation.Interface) for ind in prev_gen])
            and all([isinstance(ind, Mutation.Interface) for ind in descendants[-1]])):
      raise Exception('At least one individual cannot be mutated')
    return descendants[-1] + [ind.mutate(self.__p) for ind in [sc.obj for sc in sorted(
      [sortingClass(obj=p, cmp=self.cmp) for p in prev_gen], reverse=True)][:self.__n]]


class DeleteNLast(SortingBasedReplacement):
  """
  Delete N Last
  """
  arg_N = 'n'

  def __init__(self, **kwargs):
    super(DeleteNLast, self).__init__(**kwargs)
    self.__n = kwargs.get(self.arg_N, 4)

  def __str__(self):
    return 'DNL n=%i' % self.__n

  def new_generation(self, prev_gen, descendants):
    return [sc.obj for sc in sorted([sortingClass(obj=p, cmp=self.cmp) for p in prev_gen], reverse=True)][:-self.__n] + \
           np.random.choice(descendants[-1], self.__n, replace=False).tolist()


class DeleteN(ReplacementSchemeInterface):
  """
  Delete N
  """
  arg_N = 'n'

  def __init__(self, **kwargs):
    super(DeleteN, self).__init__(**kwargs)
    self.__n = kwargs.get(self.arg_N, 4)

  def __str__(self):
    return 'DN n=%i' % self.__n

  def new_generation(self, prev_gen, descendants):
    blacklist = np.random.choice(prev_gen, self.__n, replace=False)
    return [ind for ind in prev_gen if ind not in blacklist] + \
           np.random.choice(descendants[-1], self.__n, replace=False).tolist()


class TournamentReplacement(ReplacementSchemeInterface):
  """
  Tournament Replacement
  """

  def __init__(self, **kwargs):
    super(TournamentReplacement, self).__init__(**kwargs)
    raise Exception('Not Implemented!!')
    pass

  def new_generation(self, prev_gen, descendants):
    pass

  pass
