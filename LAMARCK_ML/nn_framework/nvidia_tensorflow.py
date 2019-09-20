import os

from LAMARCK_ML.architectures.losses import *
from LAMARCK_ML.architectures.variables.initializer import *
from LAMARCK_ML.architectures.variables.regularisation import *
from LAMARCK_ML.data_util import DimNames, IOLabel
from LAMARCK_ML.data_util.dataType import *
from LAMARCK_ML.individuals import ClassifierIndividual
from LAMARCK_ML.metrics import Accuracy, FlOps, Parameters, TimeMetric, MemoryMetric
from LAMARCK_ML.nn_framework import NeuralNetworkFrameworkInterface
from LAMARCK_ML.architectures.functions import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils


class NVIDIATensorFlow(NeuralNetworkFrameworkInterface,
                       Accuracy.Interface,
                       FlOps.Interface,
                       Parameters.Interface,
                       TimeMetric.Interface,
                       MemoryMetric.Interface):
  arg_SESSION_CFG = 'session_cfg'
  arg_BATCH_SIZE = 'batch_size'

  mapping_dtype = {
    DHalf: tf.float16,
    DFloat: tf.float32,
    DDouble: tf.float64,
    DInt8: tf.int8,
    DInt16: tf.int16,
    DInt32: tf.int32,
    DInt64: tf.int64,
    DUInt8: tf.uint8,
    DUInt16: tf.uint16,
    DUInt32: tf.uint32,
    DUInt64: tf.uint64,
  }

  mapping_initializer = {
    GlorotUniform: tf.keras.initializers.glorot_uniform,
    GlorotNormal: tf.keras.initializers.glorot_normal,
    Constant: tf.keras.initializers.constant,
    Zeros: tf.keras.initializers.zeros,
    Ones: tf.keras.initializers.ones,
  }

  mapping_regularizer = {
    NoRegularisation: None,
    L1: tf.keras.regularizers.l1,
    L2: tf.keras.regularizers.l2,
    L1L2: tf.keras.regularizers.l1_l2,
  }

  mapping_loss = {
    SoftmaxCrossEntropyWithLogits: tf.keras.losses.CategoricalCrossentropy,
    SparseSoftmaxCrossEntropyWithLogits: tf.keras.losses.SparseCategoricalCrossentropy,
    BinaryCrossentropy: tf.keras.losses.BinaryCrossentropy,
    MeanSquaredError: tf.keras.losses.MeanSquaredError,
    MeanAbsoluteError: tf.keras.losses.MeanAbsoluteError,
  }

  def __init__(self, **kwargs):
    super(NVIDIATensorFlow, self).__init__(**kwargs)
    self._sess_cfg = kwargs.get(self.arg_SESSION_CFG)
    self.batch_size = kwargs.get(self.arg_BATCH_SIZE, 32)

    self._memory = None
    self._time = None

  def setup_individual(self, individual: ClassifierIndividual):
    self._memory = None
    self._time = None
    self.sess = tf.compat.v1.Session(config=self._sess_cfg)
    K.set_session(self.sess)
    id2tfObj = dict()
    self.inputs = list()
    for ds in self.data_sets:
      for output_id_name, output in ds.outputs.items():
        if output_id_name != IOLabel.DATA:
          continue
        shape = tuple([dim.size for dim in output.shape.dim if dim.name != DimNames.BATCH])
        batch_size = output.shape[DimNames.BATCH]
        name = ds.id_name + '_' + output_id_name
        dtype = self.mapping_dtype.get(output.dtype)
        tfObj = tf.keras.Input(shape=shape, batch_size=batch_size, name=name, dtype=dtype)
        id2tfObj[ds.id_name] = {**id2tfObj.get(ds.id_name, dict()), **{output_id_name: tfObj}}
        self.inputs.append(tfObj)

    functionStack = []
    for network in individual._networks:
      functionStack.extend(network.functions)

    while functionStack:
      _func = functionStack.pop(0)
      all_found = True
      func_inputs = dict()
      for _input, out_mapping in _func.inputs.items():
        out_dict = id2tfObj.get(out_mapping[1])
        if out_dict is None or out_dict.get(out_mapping[0]) is None:
          all_found = False
          break
        func_inputs[_input] = out_dict.get(out_mapping[0])
      if not all_found:
        functionStack.append(_func)
        continue
      id2tfObj[_func.id_name] = getattr(self, _func.__class__.__name__ + '__')(_func, **func_inputs)

    self.outputs = list()
    for label, id_name in individual.network.output_mapping.values():
      out_dict = id2tfObj.get(id_name)
      if out_dict is not None and out_dict.get(label) is not None:
        tfObj = out_dict.get(label)
        tfObj = tf.keras.layers.Softmax()(tfObj)
        self.outputs.append(tfObj)

    # print(self.inputs)
    # print(self.outputs)
    self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=self.mapping_loss.get(individual.loss.__class__)(),
      metrics=['accuracy']
    )
    self.data_sets[0]('train')
    # print('data_X', self.data_sets[0].data_X)
    # print('data_Y', self.data_sets[0].data_Y)
    self.model.fit(self.data_sets[0].data_X, self.data_sets[0].data_Y,
                   batch_size=self.batch_size,
                   epochs=20,
                   verbose=0)

  def teardown_individual(self):
    self._memory = None
    self._time = None
    del self.model
    K.clear_session()
    tf.compat.v1.reset_default_graph()

  def Dense__(self, func, **kwargs):
    class C_Dense(tf.keras.layers.Dense):
      def __init__(self, kernel_trainable=True, bias_trainable=True, **kwargs):
        super(C_Dense, self).__init__(**kwargs)
        self.kernel_trainable = kernel_trainable
        self.bias_trainable = bias_trainable

      def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
          raise TypeError('Unable to build `Dense` layer with non-floating point '
                          'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel = self.add_weight(
          'kernel',
          shape=[last_dim, self.units],
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          dtype=self.dtype,
          trainable=self.kernel_trainable)
        if self.use_bias:
          self.bias = self.add_weight(
            'bias',
            shape=[self.units, ],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=self.bias_trainable)
        else:
          self.bias = None
        self.built = True

    kernel = [var for var in func.variables if var.name.endswith('kernel')][0]
    bias = [var for var in func.variables if var.name.endswith('bias')][0]
    units = func.attr.get(func.arg_UNITS)
    kernel_init = self.mapping_initializer.get(kernel.initializer.__class__)()
    if kernel.value is not None:
      kernel_init = tf.keras.initializers.Constant(value=kernel.value)
    bias_init = self.mapping_initializer.get(bias.initializer.__class__)()
    if bias.value is not None:
      bias_init = tf.keras.initializers.Constant(value=bias.value)

    kernel_reg = self.mapping_regularizer.get(kernel.regularisation.__class__)
    bias_reg = self.mapping_regularizer.get(bias.regularisation.__class__)
    tfObj = C_Dense(units=units, activation=None, use_bias=True,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    kernel_regularizer=kernel_reg() if kernel_reg is not None else None,
                    bias_regularizer=bias_reg() if bias_reg is not None else None,
                    kernel_trainable=kernel.trainable,
                    bias_trainable=bias.trainable,
                    name=func.id_name.replace(':', '_'),
                    )
    outNTS = next(iter(func.outputs))
    func_input = next(iter(kwargs.values()))
    # TODO: only working with units not images
    return {outNTS: tfObj(func_input)}

  def Merge__(self, func, **kwargs):
    first = kwargs.get(Merge._DF_INPUTS[0])
    second = kwargs.get(Merge._DF_INPUTS[1])
    outNTS_id, outNTS = next(iter(func.outputs.items()))
    axis = [i for i, d in enumerate(outNTS.shape.dim) if d.name == DimNames.UNITS or d.name == DimNames.CHANNEL][0]
    tfObj = tf.keras.layers.Concatenate(axis=axis,
                                        name=func.id_name.replace(':', '_'),
                                        )
    return {outNTS_id: tfObj([first, second])}

  def Conv2D__(self, func, **kwargs):
    class C_Conv2D(tf.keras.layers.Conv2D):
      def __init__(self, kernel_trainable=True, bias_trainable=True, **kwargs):
        super(C_Conv2D, self).__init__(**kwargs)
        self.kernel_trainable = kernel_trainable
        self.bias_trainable = bias_trainable

      def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
          channel_axis = 1
        else:
          channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
          raise ValueError('The channel dimension of the inputs '
                           'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
          name='kernel',
          shape=kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          trainable=self.kernel_trainable,
          dtype=self.dtype)
        if self.use_bias:
          self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=self.bias_trainable,
            dtype=self.dtype)
        else:
          self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        if self.padding == 'causal':
          op_padding = 'valid'
        else:
          op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
          op_padding = op_padding.upper()
        self._convolution_op = nn_ops.Convolution(
          input_shape,
          filter_shape=self.kernel.shape,
          dilation_rate=self.dilation_rate,
          strides=self.strides,
          padding=op_padding,
          data_format=conv_utils.convert_data_format(self.data_format,
                                                     self.rank + 2))
        self.built = True

    kernel = [var for var in func.variables if var.name.endswith('kernel')][0]
    bias = [var for var in func.variables if var.name.endswith('bias')][0]
    kernel_init = self.mapping_initializer.get(kernel.initializer.__class__)()
    if kernel.value is not None:
      kernel_init = tf.keras.initializers.Constant(value=kernel.value)
    bias_init = self.mapping_initializer.get(bias.initializer.__class__)()
    if bias.value is not None:
      bias_init = tf.keras.initializers.Constant(value=bias.value)

    kernel_reg = self.mapping_regularizer.get(kernel.regularisation.__class__)
    bias_reg = self.mapping_regularizer.get(bias.regularisation.__class__)

    tfObj = tf.keras.layers.Conv2D(filters=func.attr.get(func.arg_FILTER),
                                   kernel_size=(func.attr.get(func.arg_KERNEL_HEIGHT),
                                                func.attr.get(func.arg_KERNEL_WIDTH)),
                                   strides=(func.attr.get(func.arg_STRIDE_HEIGHT),
                                            func.attr.get(func.arg_STRIDE_WIDTH)),
                                   padding=func.attr.get(func.arg_PADDING),
                                   use_bias=True,
                                   kernel_initializer=kernel_init,
                                   bias_initializer=bias_init,
                                   kernel_regularizer=kernel_reg,
                                   bias_regularizer=bias_reg,
                                   )

    outNTS = next(iter(func.outputs))
    func_input = next(iter(kwargs.values()))
    return {outNTS.id_name: tfObj(func_input)}

  def _time_memory_flops_params(self):
    self.data_sets[0].batch = 1
    random_input = next(iter(self.data_sets[0])).get(IOLabel.DATA)
    run_meta = tf.compat.v1.RunMetadata()
    self.sess.run(self.outputs[0].name, feed_dict={self.inputs[0]: random_input},
                  options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                  run_metadata=run_meta)
    options = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory(min_micros=0, min_bytes=0, )
    options['output'] = 'none'
    options['verbose'] = 0
    time_mem = tf.compat.v1.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=options, )

    self._parameters = time_mem.total_parameters
    self._flops = time_mem.total_float_ops
    self._time = time_mem.total_exec_micros
    self._memory = time_mem.total_peak_bytes

  def time(self):
    if self._time is None:
      self._time_memory_flops_params()
    return self._time

  def memory(self):
    if self._memory is None:
      self._time_memory_flops_params()
    return self._memory

  def accuracy(self):
    for ds in self.data_sets:
      ds('train')
    _, acc = self.model.evaluate(self.data_sets[0].data_X,
                                 self.data_sets[0].data_Y,
                                 batch_size=self.batch_size,
                                 verbose=0)
    return acc

  def flops_per_sample(self):
    if self._flops is None:
      self._time_memory_flops_params()
    return self._flops

  def parameters(self):
    if self._parameters is None:
      self._time_memory_flops_params()
    return self._parameters

  pass
