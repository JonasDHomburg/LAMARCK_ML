from random import choice, random, sample, randint
from typing import Type, Set, Dict, Tuple, List
from datetime import datetime
from functools import partial

import numpy as np

from LAMARCK_ML.architectures import DataFlow
from LAMARCK_ML.architectures.IOMapping_pb2 import IOMappingProto
from LAMARCK_ML.architectures.NeuralNetwork_pb2 import NeuralNetworkProto
from LAMARCK_ML.architectures.functions import Function, InvalidFunctionType
from LAMARCK_ML.architectures.interface import ArchitectureInterface
from LAMARCK_ML.data_util import TypeShape
from LAMARCK_ML.data_util.attribute import attr2pb, pb2attr, value2pb, pb2val
from LAMARCK_ML.reproduction.methods import Mutation, Recombination, RandomStep


class WeightAgnosticNeuralNetwork(ArchitectureInterface, DataFlow,
                                  Mutation.Interface,
                                  RandomStep.Interface,
                                  Recombination.Interface,
                                  ):
  arg_INPUTS = 'inputs'
  arg_OUTPUT_TARGETS = 'output_targets'
  arg_FUNCTIONS = 'functions'
  arg_INITIAL_DEPTH = 'initial_depth'

  attr_nodes = 'nodes'

  def __init__(self, **kwargs):
    super(WeightAgnosticNeuralNetwork, self).__init__(**kwargs)
    self.attr = dict()
    self.input_mapping = kwargs.get(self.arg_INPUTS, {})
    self.output_targets = kwargs.get(self.arg_OUTPUT_TARGETS, {})
    self.function_cls = kwargs.get(self.arg_FUNCTIONS)
    try:
      if not (isinstance(self.function_cls, list) and all(
          [isinstance(f, type) and issubclass(f, Function) for f in self.function_cls])):
        raise InvalidFunctionType("Invalid function type in 'functions'!")
    except Exception:
      raise InvalidFunctionType("Invalid function type in 'functions'!")

    self.attr[self.attr_nodes], self.functions, self.output_mapping = self.random_network_init_single_output(
      perceptron_pool=set(self.function_cls),
      input_mapping=self.input_mapping,
      output_targets=self.output_targets
    )

    self._inputs = {key: (out_label, df_obj_id) for key, (out_label, _, df_obj_id) in self.input_mapping.items()}
    self._id_name = self.getNewName()

  @staticmethod
  def random_network_init_single_output(perceptron_pool: Set[Type[Function]] = None,
                                        input_mapping: Dict[str, Tuple[str, TypeShape, str]] = None,
                                        output_targets: Dict[str, TypeShape] = None,
                                        initial_depth=1,
                                        ):
    nodes = {key: (0, {out_label: ts}) for key, (out_label, ts, _) in input_mapping.items()}
    out_node_label, out_node_ts = sample(output_targets.items(), k=1)[0]
    node_instances = list()
    max_depth = 0
    while True:
      # might stuck for ever if malicious input is provided
      perc_cls = sample(perceptron_pool, k=1)[0]
      in_node = sample(nodes.items(), k=1)[0]
      in_id, (in_depth, in_outputs) = in_node
      for other_inputs, outputs, out_label in perc_cls.possible_output_shapes(
          input_ntss=in_outputs,
          target_output=out_node_ts,
          is_reachable=partial(WeightAgnosticNeuralNetwork.reachable,
                               max_depth=initial_depth - in_depth,
                               function_pool=perceptron_pool)
      ):
        n_depth = in_depth
        used_nodes = {il: (ol, in_outputs, in_id) for il, ol in out_label.items()}
        available_nodes = set(nodes.keys()).difference({in_id})
        all_found = True
        for o_label, o_ts in other_inputs.items():
          try:
            pn_id, un_key, un_value = sample([(n, o_label, (l, nodes[n][1], n))
                                              for n in available_nodes
                                              if nodes[n][0] <= in_depth
                                              for l, ts in nodes[n][1].items()
                                              if ts == o_ts
                                              ],
                                             k=1)[0]
          except Exception:
            all_found = False
            break
          available_nodes.remove(pn_id)
          n_depth = max(n_depth, nodes[pn_id][0])
          used_nodes[un_key] = un_value
        if not all_found:
          continue

        parameters, possibilities = perc_cls.generateParameters(
          input_dict=used_nodes,
          expected_outputs=outputs,
        )
        node = perc_cls(**np.random.choice(
          parameters,
          size=1,
          replace=False,
          p=possibilities
        )[0])
        node_instances.append(node)

        n_depth += 1
        nodes[node.id_name] = (n_depth, node.outputs)
        max_depth = max(n_depth, max_depth)
        break
      if max_depth >= initial_depth:
        break
    return nodes, node_instances, {out_node_label: (next(iter(outputs.keys())), node.id_name)}

  @staticmethod
  def wann_initialization(perceptron_pool: Set[Type[Function]] = None,
                          input_mapping: Dict[str, Tuple[str, TypeShape, str]] = None,
                          output_targets: Dict[str, TypeShape] = None,
                          ):
    # TODO: implement
    pass

  @classmethod
  def getNewName(cls):
    return cls.__name__ + '_' + str(datetime.now().timestamp()) + '_%09i' % randint(0, 1e9 - 1)

  @property
  def outputs(self) -> Set[TypeShape]:
    return self.output_targets

  def inputs(self) -> Dict[str, Tuple[str, str]]:
    return self._inputs

  @property
  def id_name(self) -> str:
    return self._id_name

  def __eq__(self, other):
    if (isinstance(other, self.__class__)
        and self.id_name == other.id_name
        and len(self.function_cls) == len(other.function_cls) == len(
          [f for f in self.function_cls if f in other.function_cls])
        and len(self._inputs) == len(other._inputs) == len(
          [self._inputs.get(key) == other._inputs.get(key) for key in self._inputs])
        and len(self.input_mapping) == len(other.input_mapping) == len(
          {k: None for k in self.input_mapping if self.input_mapping.get(k) == other.input_mapping.get(k)})
        and len(self.output_targets) == len(other.output_targets)
        and all([any([ot == st for ot in other.output_targets]) for st in self.output_targets])
        and len(self.functions) == len(other.functions) == len(
          [True for f in self.functions if f in other.functions])
        and len(self.attr) == len(other.attr) == len(
          {k: None for k in self.attr if self.attr.get(k) == other.attr.get(k)})
        and len(self.output_mapping) == len(other.output_mapping) == len(
          [True for om in self.output_mapping if om in other.output_mapping])
    ):
      return True
    return False

  def __getstate__(self):
    return self.get_pb().SerializeToString()

  def get_pb(self, result=None):
    if not isinstance(result, NeuralNetworkProto):
      result = NeuralNetworkProto()
    result.id_name = self._id_name
    for f_cls in self.function_cls:
      result.function_cls.append(f_cls.get_cls_name())
    for _f in self.functions:
      result.functions.append(_f.get_pb())
    for in_label, (out_label, id_name) in self._inputs.items():
      ioM = IOMappingProto()
      ioM.in_label = in_label
      ioM.out_label = out_label
      ioM.df_id_name = id_name
      result.input_mapping.append(ioM)
    for in_label, (out_label, id_name) in self.output_mapping.items():
      ioM = IOMappingProto()
      ioM.in_label = in_label
      ioM.out_label = out_label
      ioM.df_id_name = id_name
      result.output_mapping.append(ioM)
    result.output_ntss.v.extend([value2pb(v) for v in self.output_targets.items()])

    result.attr.extend([attr2pb(key, value) for key, value in self.attr.items()])
    result.attr.append(attr2pb(self.arg_INPUTS, self.input_mapping))
    return result

  def __setstate__(self, state):
    def build_function(pb):
      result = Function.__new__(Function)
      result.__setstate__(pb)
      return result

    if isinstance(state, str) or isinstance(state, bytes):
      _nn = NeuralNetworkProto()
      _nn.ParseFromString(state)
    elif isinstance(state, NeuralNetworkProto):
      _nn = state
    else:
      return
    self._id_name = _nn.id_name
    self.output_targets = dict([pb2val(v) for v in _nn.output_ntss.v])
    self.function_cls = [Function.getClassByName(f_cls) for f_cls in _nn.function_cls]
    self._inputs = dict([(ioMP.in_label, (ioMP.out_label, ioMP.df_id_name)) for ioMP in _nn.input_mapping])
    self.output_mapping = dict([(ioMP.in_label, (ioMP.out_label, ioMP.df_id_name)) for ioMP in _nn.output_mapping])

    self.functions = [build_function(_f) for _f in _nn.functions]
    self.attr = dict([pb2attr(attr) for attr in _nn.attr])
    self.input_mapping = self.attr.pop(self.arg_INPUTS)

  def norm(self, other):
    # TODO: implement
    pass

  def __sub__(self, other):
    # TODO: implement
    pass

  def update_state(self, *args, **kwargs):
    # TODO: store weights
    pass

  @staticmethod
  def __insert_node(wann):
    result = wann.__copy__()
    result._id_name = wann.__class__.getNewName()
    nodes = result.attr[result.attr_nodes]
    possible_nodes = list(nodes.keys())
    connect_node = None
    while True:
      node_id = choice(possible_nodes)
      possible_connect_nodes = [n for n in result.functions if node_id in [_id for _, _id in n.inputs.values()]] \
                               + [n for n in result.outputs
                                  if n not in result.output_mapping or
                                  result.output_mapping[n][1] == node_id]
      if len(possible_connect_nodes) > 0:
        connect_node = choice(possible_connect_nodes)
        break
    if connect_node is None:
      # => inserting a node is not guaranteed
      return result
    node_depth, node_outputs = nodes[node_id]
    node_pool = list(result.function_cls)
    target_typeshape = choice(list(node_outputs.values()))

    node_not_inserted = True
    while len(node_pool) and node_not_inserted:
      node_type = sample(node_pool, k=1)[0]
      node_pool.remove(node_type)
      for other_inputs, outputs, in_label_mapping in node_type.possible_output_shapes(
          input_ntss=node_outputs,
          target_output=target_typeshape,
          is_reachable=partial(WeightAgnosticNeuralNetwork.reachable, max_depth=1, function_pool=result.function_cls),
          max_inputs=1,
      ):
        if len(other_inputs) > 0:
          continue
        parameters, possibilities = node_type.generateParameters(
          input_dict={il: (ol, node_outputs, node_id) for il, ol in in_label_mapping.items()},
          expected_outputs=outputs,
        )
        new_node = node_type(**np.random.choice(
          parameters,
          size=1,
          replace=False,
          p=possibilities,
        )[0])
        nodes[new_node.id_name] = (node_depth + 1, new_node.outputs)
        result.functions.append(new_node)
        if isinstance(connect_node, str):
          result.output_mapping[connect_node] = (choice([label for label, ts in new_node.outputs.items()
                                                         if ts == target_typeshape]), new_node.id_name)
        else:
          for key, (label, _id) in connect_node.inputs.items():
            if _id == node_id:
              connect_node.inputs[key] = (label, new_node.id_name)
              break
          dependencies = dict()
          id_name_to_function = dict()
          for f in result.functions:
            for _, in_id in f.inputs.values():
              dependencies[in_id] = dependencies.get(in_id, []) + [f.id_name]
            id_name_to_function[f.id_name] = f
          stack = list(nodes.keys())
          updated = set()
          while stack:
            update_node_id = stack.pop(0)
            update_node = id_name_to_function.get(update_node_id)
            if update_node is None:
              updated.add(update_node_id)
              continue
            if all([in_id in updated for _, in_id in update_node.inputs.values()]):
              nodes[update_node_id] = (
              max([nodes[in_id][0] for _, in_id in update_node.inputs.values()]) + 1,
              update_node.outputs)
              updated.add(update_node_id)
              continue
            stack.append(update_node_id)
        node_not_inserted = False
    return result

  @staticmethod
  def __add_connection(wann):
    def creates_cycle(node, target, depth, id_to_obj):
      check_ancestor = [id_to_obj[target]]
      node_id = node.id_name
      for _ in range(depth):
        _ca = list()
        for n in check_ancestor:
          for _, (_, in_id) in n.inputs.items():
            if in_id == node_id:
              return True
            if in_id in id_to_obj:
              _ca.append(id_to_obj[in_id])
        check_ancestor = _ca
      return False

    result = wann.__copy__()
    result._id_name = wann.__class__.getNewName()
    node = choice(result.functions)
    result_nodes = result.attr[wann.attr_nodes]
    node_depth = result_nodes[node.id_name][0]
    node_id_to_obj = {n.id_name: n for n in result.functions}
    pool = list(result_nodes.items())
    while True:
      c = choice(range(len(pool)))
      new_in, (depth, _) = pool.pop(c)
      if new_in != node.id_name and \
          (new_in not in node_id_to_obj or
           not creates_cycle(node, new_in, depth - node_depth, node_id_to_obj)):
        break
    n_node = node.add_input(new_in, result_nodes[new_in][1])
    result.functions.remove(node)
    for n in result.functions:
      n.input_mapping = {in_label: (out_label, n_id if n_id != node.id_name else n_node.id_name)
                         for in_label, (out_label, n_id) in n.input_mapping.items()}
    result.functions.append(n_node)
    result_nodes[n_node.id_name] = (result_nodes.pop(node.id_name)[0], n_node.outputs)

    stack = list(result_nodes.keys())
    updated = set()
    while stack:
      update_node_id = stack.pop(0)
      update_node = node_id_to_obj.get(update_node_id)
      if update_node is None:
        updated.add(update_node_id)
        continue
      if all([in_id in updated for _,in_id in update_node.inputs.values()]):
        result_nodes[update_node_id] = (max([result_nodes[in_id][0] for _, in_id in update_node.inputs.values()])+1,
                                        update_node.outputs)
        updated.add(update_node_id)
        continue
      stack.append(update_node_id)
    return result

  @staticmethod
  def __mutate_node(wann):
    result = wann.__copy__()
    result._id_name = wann.__class__.getNewName()
    node = choice(result.functions)
    n_node = node.mutate(1)
    result.functions.remove(node)
    for n in result.functions:
      n.input_mapping = {in_label: (out_label, n_id if n_id != node.id_name else n_node.id_name)
                         for in_label, (out_label, n_id) in n.input_mapping.items()}
    result.functions.append(n_node)
    result.output_mapping = {out_label: (label, n_id if n_id != node.id_name else n_node.id_name)
                             for out_label, (label, n_id) in result.output_mapping.items()}
    result.attr[wann.attr_nodes][n_node.id_name] = (result.attr[wann.attr_nodes].pop(node.id_name)[0], n_node.outputs)
    return result

  def __copy__(self):
    result = WeightAgnosticNeuralNetwork.__new__(WeightAgnosticNeuralNetwork)
    result.input_mapping = {key: (l0, v.__copy__(), l1)
                            for key, (l0, v, l1) in self.input_mapping.items()}
    result.output_targets = {key: ts.__copy__()
                             for key, ts in self.output_targets.items()}
    result.function_cls = list(self.function_cls)
    result._inputs = {k: (v0, v1) for k, (v0, v1) in self._inputs.items()}
    result.output_mapping = {k: (v0, v1) for k, (v0, v1) in self.output_mapping.items()}
    result.functions = [f.__copy__() for f in self.functions]
    result.attr = dict([pb2attr(attr2pb(key, value)) for key, value in self.attr.items()])

    self._id_name = self.getNewName()
    return result

  def mutate(self, prob):
    if random() < prob:
      result = [choice([
        WeightAgnosticNeuralNetwork.__insert_node,
        WeightAgnosticNeuralNetwork.__add_connection,
        WeightAgnosticNeuralNetwork.__mutate_node
      ])(self)]
    else:
      result = [self.__copy__()]
    return result

  def step(self, step_size):
    result = self
    for _ in range(step_size):
      result = choice([
        WeightAgnosticNeuralNetwork.__insert_node,
        WeightAgnosticNeuralNetwork.__add_connection,
        WeightAgnosticNeuralNetwork.__mutate_node
      ])(result)
    return [result]

  def recombine(self, other):
    def add_functions_nodes(result, dependencies, id_name_to_function, outputs, parent_nodes):
      resolve_dependencies = list(outputs)
      nodes = result.attr[result.attr_nodes]
      while resolve_dependencies:
        function_id = resolve_dependencies.pop(0)
        f_dependencies = dependencies.get(function_id)
        if function_id in parent_nodes:
          nodes[function_id] = parent_nodes[function_id]
        if f_dependencies:
          resolve_dependencies.extend(f_dependencies)
          result.functions.append(id_name_to_function[function_id])

    def construct_result(parent, self_outputs, other_outputs):
      result = parent.__copy__()
      result.output_mapping = {**{output: self.output_mapping[output]
                                  for output in self_outputs if output in self.output_mapping},
                               **{output: other.output_mapping[output]
                                  for output in other_outputs if output in other.output_mapping}}
      result.functions = list()
      result.attr[result.attr_nodes] = dict()
      return result

    outputs = list(self.outputs.keys())
    bool_vec = [randint(0, 1) for _ in range(len(outputs))]
    self_outputs = [o for o, b in zip(outputs, bool_vec) if b == 1]
    other_outputs = [o for o, b in zip(outputs, bool_vec) if b == 0]

    self_dependencies = dict()
    self_id_name_to_function = dict()
    for f in self.functions:
      for _, in_id in f.inputs.values():
        self_dependencies[in_id] = self_dependencies.get(in_id, []) + [f.id_name]
      self_id_name_to_function[f.id_name] = f

    other_dependencies = dict()
    other_id_name_to_function = dict()
    for f in other.functions:
      for _, in_id in f.inputs.values():
        other_dependencies[in_id] = other_dependencies.get(in_id, []) + [f.id_name]
      other_id_name_to_function[f.id_name] = f

    resultA = construct_result(self, self_outputs, other_outputs)
    add_functions_nodes(resultA, self_dependencies, self_id_name_to_function, self_outputs,
                        self.attr[self.attr_nodes])
    add_functions_nodes(resultA, other_dependencies, other_id_name_to_function, other_outputs,
                        other.attr[other.attr_nodes])
    resultB = construct_result(other, other_outputs, self_outputs)
    add_functions_nodes(resultB, self_dependencies, self_id_name_to_function, other_outputs,
                        self.attr[self.attr_nodes])
    add_functions_nodes(resultB, other_dependencies, other_id_name_to_function, self_outputs,
                        other.attr[other.attr_nodes])
    return [resultA, resultB]

  @staticmethod
  def reachable(
      input_nts: TypeShape,
      target_nts: TypeShape,
      max_depth: int,
      function_pool: set,
  ):
    class FoundException(Exception):
      pass

    if input_nts.__cmp__(target_nts) == -1 and target_nts.__cmp__(input_nts) == -1:
      return True

    low_list, high_list = [input_nts], [input_nts]
    try:
      for _ in range(max_depth):
        n_low_list = list(low_list)
        for low_ in low_list:
          for func in function_pool:
            _min = func.min_transform(low_)
            if _min is None:
              continue
            if _min.__cmp__(target_nts) == -1:
              raise FoundException()
            new_nts = True
            for curr in n_low_list:
              cmp = _min.__cmp__(curr)
              if cmp == -1:
                n_low_list.remove(curr)
                n_low_list.append(_min)
                new_nts = False
                break
              if cmp == 0:
                continue
              break
            if new_nts:
              n_low_list.append(_min)
        low_list = n_low_list
    except Exception as e:
      if not isinstance(e, FoundException):
        return False
      try:
        for i in range(max_depth):
          n_high_list = list(high_list)
          for high_ in high_list:
            for func in function_pool:
              _max = func.max_transform(high_)
              if _max is None:
                continue
              if target_nts.__cmp__(_max) == -1:
                raise FoundException()
              new_nts = True
              for curr in n_high_list:
                cmp = curr.__cmp__(_max)
                if cmp == -1:
                  n_high_list.remove(curr)
                  n_high_list.append(_max)
                  new_nts = False
                  break
                if cmp == 0:
                  continue
                break
              if new_nts:
                n_high_list.append(_max)
          high_list = n_high_list
      except Exception as e:
        if not isinstance(e, FoundException):
          return False
        return True
    return False

  @property
  def inputLabels(self) -> List[str]:
    return list(self._inputs.keys())
