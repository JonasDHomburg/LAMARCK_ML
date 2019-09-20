class sortingClass(object):
  def __init__(self, obj, cmp=None):
    self.obj = obj
    self.cmp = cmp
    if self.cmp is None:
      self.cmp = lambda x, y: 0 if x == y else -1 if x < y else 1

  def __gt__(self, other):
    return self.cmp(self.obj, other.obj) > 0
