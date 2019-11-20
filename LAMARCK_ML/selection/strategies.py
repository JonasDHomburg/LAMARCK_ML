import math

import numpy as np

from LAMARCK_ML.utils.sortingClass import SortingClass
from LAMARCK_ML.selection.interface import SelectionStrategyInterface


class SortingBasedSelection(SelectionStrategyInterface):
  arg_CMP = 'cmp'

  def __init__(self, **kwargs):
    super(SortingBasedSelection, self).__init__(**kwargs)
    self.cmp = kwargs.get(self.arg_CMP)


class LinearRankingSelection(SortingBasedSelection):
  """
  Takes the individuals according to their arrangement in the population in increasing or decreasing order of fitness.
  The linear ranking scheme determines the selection probability of an individual i with the following function:
  Pr_i = \frac{1}{\Pi}(\alpha + (\beta - \alpha)\frac{rank(i)-1}{\Pi-1})
  Where the best individual i is assigned rank(i)=1 in a population with size \Pi

  When the conditions \alpha+\beta = 2 and 1 \le \alpha \le 2 are satisfied, the best individual will produce no more
  than twice offspring than the population average.

  This class performs a linear ranking!
  """
  arg_ALPHA = 'alpha'
  arg_BETA = 'beta'
  arg_R = 'r'

  def __init__(self, **kwargs):
    """
    :param alpha: Proportion for selecting worst individual
    :param beta: Proportion for selecting best individual
    :param r: If Provided alpha and beta will be overwritten to \alpha=2/(r+1) and \beta=2r/(r+1)
    """
    super(LinearRankingSelection, self).__init__(**kwargs)
    self._alpha = kwargs.get(self.arg_ALPHA, 1.5)
    self._beta = kwargs.get(self.arg_BETA, 2 - self._alpha)
    if self.arg_R in kwargs:
      r = kwargs.get(self.arg_R)
      self._alpha = 2 / (r + 1)
      self._beta = 2 * r / (r + 1)

  def __str__(self):
    return "LRS"

  def select(self, pool):
    sorted_pool = [sc.obj for sc in sorted([SortingClass(obj=p, cmp=self.cmp) for p in pool], reverse=True)]
    # pi_ is not \Pi of the description! pi_ is actually \Pi-1
    pi_ = len(pool) - 1
    # Since the sum of the probabilities is normalized anyways the constant factor of 1/pi is ignored.
    prob = np.asarray([self._alpha + (self._beta - self._alpha) * i / pi_ for i in range(pi_ + 1)])
    prob = prob / (pi_ + 1)
    prob = prob / prob.sum()
    return np.random.choice(sorted_pool, self._limit, replace=False, p=prob).tolist()


pass


class ExponentialRankingSelection(SortingBasedSelection):
  """
  'Exponential' ranking selection
  Probability p_r of an individual with rank r \elem \mathbb(N) in [0,n] and n = |pool| is given by:
  p_r = (1-p_c)^r * p_c for r < n
  and
  p_r = (1-p_c)^r for r = n
  with p_c \elem \mathbb{R} in [0,1]

  first found:
    https://youtu.be/kHyNqSnzP8Y?t=1255
  alternative source:
    https://www.tik.ee.ethz.ch/file/6c0e384dceb283cd4301339a895b72b8/TIK-Report11.pdf

  This class performs an exponential ranking!
  """
  arg_P = 'p'

  def __init__(self, **kwargs):
    """
    :param p: Probability p_c to select individual with rank 0
    """
    super(ExponentialRankingSelection, self).__init__(**kwargs)
    self._p_c = kwargs.get(self.arg_P, 0.4)

  def __str__(self):
    return "RSV1"

  def select(self, pool):
    sorted_pool = [sc.obj for sc in sorted([SortingClass(obj=p, cmp=self.cmp) for p in pool], reverse=True)]

    prob = np.asarray([((1 - self._p_c) ** i) * self._p_c for i in range(len(pool))])
    prob[-1] = prob[-1] / self._p_c
    prob = prob / prob.sum()
    return np.random.choice(sorted_pool, self._limit, replace=False, p=prob).tolist()

  pass


class MaxDiversitySelection(SelectionStrategyInterface):
  """
  maximize fitness and diversity
  difference/distance to previously selected individuals is accumulated both are normalized [0,1]
  selects individual which maximizes f^2 + d^2*c with weighting factor c, fitness f and normalized distance d to all
  previously selected individuals

  source: https://youtu.be/kHyNqSnzP8Y?t=1640
  """
  arg_DIVERSITY = 'diversity'

  def __init__(self, **kwargs):
    """
    :param diversity: Weighting factor c for diversity
    """
    super(MaxDiversitySelection, self).__init__(**kwargs)
    self._diversity_weight = kwargs.get(self.arg_DIVERSITY, 1.0)

  def __str__(self):
    return "RWMDS"

  def select(self, pool):
    result = []

    min_f, max_f = min([p.fitness for p in pool]), max([p.fitness for p in pool])
    max_f -= (min_f - 0.01)
    tmp_pool = [(p, (p.fitness - min_f) / (max_f), 0, 0) for p in pool]

    for _ in range(self._limit):
      p_max = max(tmp_pool, key=lambda x: x[1] * x[1] + x[3] * x[3] * self._diversity_weight)
      tmp_pool.remove(p_max)
      result.append(p_max[0])
      distances = [(p_max[0] - p[0]) + p[2] for p in tmp_pool]
      min_d = min(distances)
      max_d = max(distances) - min_d + .01
      tmp_pool = [(p[0], p[1], d, (d - min_d + .01) / max_d) for d, p in zip(distances, tmp_pool)]
    return result

  pass


class ProportionalSelection(SelectionStrategyInterface):
  """
  "Roulette-Wheel Selection" or "Fitness-Proportional Selection"
  Probability p_i of individual i with fitness f_i to be selected is given by:
  p_i = f_i/sum(f_i)
  """

  def __init__(self, **kwargs):
    super(ProportionalSelection, self).__init__(**kwargs)

  def __str__(self):
    return "RWS"

  def select(self, pool):
    prob = np.asarray([individual.fitness for individual in pool])
    prob = prob / prob.sum()
    return np.random.choice(pool, self._limit, replace=False, p=prob).tolist()

  pass


class TournamentSelection(SortingBasedSelection):
  arg_ST = 'St'

  def __init__(self, **kwargs):
    super(TournamentSelection, self).__init__(**kwargs)
    self._St = kwargs.get(self.arg_ST, 3)

  def __str__(self):
    return "ToS"

  def select(self, pool):
    result = [max([SortingClass(obj=obj,
                                cmp=self.cmp) for obj in np.random.choice(pool, self._St, replace=False).tolist()]).obj
              for _ in range(self._limit)]
    return result

  pass


class TruncationSelection(SortingBasedSelection):
  """
  This scheme chooses for reproduction only from the top individuals according to their rank in the population,
  and these top individuals have the same chance to be taken. The fraction of the top individuals is defined by a rank
  threshold \mu.
  """
  arg_U = 'u'

  def __init__(self, **kwargs):
    super(TruncationSelection, self).__init__(**kwargs)
    self._u = kwargs.get(self.arg_U, .7)

  def __str__(self):
    return "TrS"

  def select(self, pool):
    sorted_pool = [sc.obj for sc in sorted([SortingClass(obj=p, cmp=self.cmp) for p in pool], reverse=True)]
    if isinstance(self._u, int):
      sorted_pool = sorted_pool[:self._u]
    elif isinstance(self._u, float):
      sorted_pool = sorted_pool[:int(self._u * len(sorted_pool))]
    return np.random.choice(sorted_pool, self._limit, replace=False).tolist()

  pass


class GreedyOverSelection(SortingBasedSelection):
  """
  Greedy over-selection is usually used when population is large, e.g. more than 1000 individuals. In this method,
  individuals are selected according to their fitness values (fitness-proportionate selection). However, this method
  biases selection towards the highest performers.
  Using the fitness values, the population is divided into two groups by \epsilon. Group I includes the top \epsilon\%
  of individuals while Group II contains the remaining.
  """

  arg_E = 'e'

  def __init__(self, **kwargs):
    """
    :param e: Percentage of individuals for Group I. Typically < 0.5.
    """
    super(GreedyOverSelection, self).__init__(**kwargs)
    self._top = kwargs.get(self.arg_E, 0.2)

  def __str__(self):
    return "GOS"

  def select(self, pool):
    # sorted_pool = sorted(pool, key=lambda x: x.fitness, reverse=True)
    sorted_pool = [sc.obj for sc in sorted([SortingClass(obj=p, cmp=self.cmp) for p in pool], reverse=True)]
    ind = int(len(sorted_pool) * self._top)
    high, low = sorted_pool[:ind], sorted_pool[ind:]
    high_limit = int(math.ceil(self._limit * .5))
    low_limit = self._limit - high_limit

    high_prob = np.asarray([i.fitness for i in high])
    high_prob = high_prob / high_prob.sum()

    low_prob = np.asarray([i.fitness for i in low])
    low_prob = low_prob / low_prob.sum()

    result = np.random.choice(high, high_limit, replace=False, p=high_prob).tolist() + \
             np.random.choice(low, low_limit, replace=False, p=low_prob).tolist()
    return result
