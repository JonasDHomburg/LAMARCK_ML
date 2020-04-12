from LAMARCK_ML.individuals.interface import IndividualInterface
from LAMARCK_ML.reproduction.methods import Mutation, Recombination, RandomStep
from LAMARCK_ML.metrics.implementations import CartesianFitness
import numpy as np


class CartesianIndividual(IndividualInterface,
                          Mutation.Interface,
                          RandomStep.Interface,
                          Recombination.Interface,
                          CartesianFitness.Interface):
  arg_Dimensions = 'dimensions'
  arg_State = 'state'
  arg_FitnessFunction = 'fitnessfunction'

  def __init__(self, **kwargs):
    super(CartesianIndividual, self).__init__(**kwargs)
    self.attr[self.arg_Dimensions] = kwargs.get(self.arg_Dimensions, 2)
    self.state = kwargs.get(self.arg_State)
    if self.state is None:
      self.state = np.random.random(self.attr[self.arg_Dimensions])
    assert isinstance(self.state, np.ndarray)
    assert self.state.shape == (self.attr[self.arg_Dimensions],)
    # elif not isinstance(self.state, np.ndarray):
    #   raise Exception('state must be an instance of numpy.ndarray')
    self.attr[self.arg_State] = self.state
    self.attr[self.arg_FitnessFunction] = kwargs.get(self.arg_FitnessFunction)
    if not isinstance(self.attr[self.arg_FitnessFunction], str):
      self.attr[self.arg_FitnessFunction] = 'np.prod((np.sin(x)+1)/2)*np.prod(np.exp(-(x-20.4)**2/500))'

  def __sub__(self, other):
    return self.state - other.state

  def norm(self, other):
    return np.linalg.norm(self - other)

  def update_state(self, *args, **kwargs):
    """
    Not implemented and used since this individual does not require a training.
    """
    pass

  def mutate(self, prob):
    d = self.attr[self.arg_Dimensions]
    new_ind = CartesianIndividual(**{
      self.arg_Dimensions: d,
      self.arg_State: np.maximum(self.state + (np.random.random(d) - 0.5) * prob, np.zeros(d)),
      self.arg_FitnessFunction: self.attr[self.arg_FitnessFunction],
    })
    return [new_ind]

  def step(self, step_size):
    return self.mutate(step_size)

  def recombine(self, other):
    dim = self.attr[self.arg_Dimensions]
    idx = np.random.randint(0, 2, size=dim)
    idx_rv = np.abs(1 - idx)
    stacked = np.stack([self.state, other.state])
    arange = np.arange(dim)
    result = [
      CartesianIndividual(**{
        self.arg_Dimensions: dim,
        self.arg_State: stacked[idx, arange],
        self.arg_FitnessFunction: self.attr[self.arg_FitnessFunction],
      }),
      CartesianIndividual(**{
        self.arg_Dimensions: dim,
        self.arg_State: stacked[idx_rv, arange],
        self.arg_FitnessFunction: self.attr[self.arg_FitnessFunction],
      })
    ]
    return result

  def cartesianFitness(self):
    x = self.state.copy()
    return float(eval(self.attr[self.arg_FitnessFunction]))

  def _cls_setstate(self, state):
    super(CartesianIndividual, self)._cls_setstate(state)
    self.state = self.attr[self.arg_State]

  @property
  def fitness(self):
    if self.metrics:
      return self.metrics[CartesianFitness.ID]
    return -.1

  def build_instance(self, nn_framework):
    pass

  def train_instance(self, nn_framework):
    pass
