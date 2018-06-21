# -*- coding: utf-8 -*-
#/usr/bin/python2

import numpy as np
import math
from functools import reduce
from operator import mul
import mxnet as mx

def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x, x_h = mx.nd.split(x, 2, axis = -1)
    return mx.ndarray.sigmoid(x) * x_h

def noam_norm(x, epsilon=1.0, scope=None, reuse=None):
    """One version of layer normalization."""
    #layer = mx.gluon.nn.LayerNorm(epsilon=epsilon)
    layer = mx.ndarray.InstanceNorm(epsilon=epsilon)

    return layer(x)

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = mx.ndarray.mean(x, axis=[-1], keepdims=True)
    #mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    #variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    variance = mx.ndarray.mean(mx.ndarray.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * mx.ndarray.sqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    layer = mx.gluon.nn.LayerNorm(epsilon=epsilon)

    return layer(x)

norm_fn = layer_norm#tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm

def highway(x, size = None, activation = None, num_layers = 2, scope = "highway", dropout = 0.0, reuse = None):
    if size is None:
        size = x.shape.as_list()[-1]
    else:
        x = conv(x, size, name="input_projection", reuse=reuse)
    for i in range(num_layers):
        T = conv(x, size, bias=True, activation="sigmoid", name="gate_%d" % i, reuse=reuse)
        H = conv(x, size, bias=True, activation=activation, name="activation_%d" % i, reuse=reuse)
        H = mx.gluon.nn.Dropout(1.0 - dropout)(H)
        x = H * T + x * (1.0 - T)
    return x

def layer_dropout(inputs, residual, dropout):
    pred = mx.nd.random_uniform() < dropout

    if pred:
        return lambda: residual
    else:
        return lambda: mx.gluon.nn.Dropout(1.0 - dropout)(inputs) + residual

    # pred = tf.random_uniform([]) < dropout
    #return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)

def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask = None,
                   num_filters = 128, input_projection = False, num_heads = 8,
                   seq_len = None, scope = "res_block", is_training = True,
                   reuse = None, bias = True, dropout = 0.0):
    if input_projection:
        inputs = conv(inputs, num_filters, name="input_projection", reuse=reuse)
    outputs = inputs
    sublayer = 1
    total_sublayers = (num_conv_layers + 2) * num_blocks
    for i in range(num_blocks):
        outputs = add_timing_signal_1d(outputs)
        outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                                       seq_len=seq_len, scope="encoder_block_%d" % i, reuse=reuse, bias=bias,
                                       dropout=dropout, sublayers=(sublayer, total_sublayers))
        outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask=mask, num_heads=num_heads,
                                                 scope="self_attention_layers%d" % i, reuse=reuse,
                                                 is_training=is_training,
                                                 bias=bias, dropout=dropout, sublayers=(sublayer, total_sublayers))
    return outputs


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len = None, scope = "conv_block", is_training = True,
               reuse = None, bias = True, dropout = 0.0, sublayers = (1, 1)):
    outputs = mx.ndarray.expand_dims(inputs, 2)
    l, L = sublayers
    for i in range(num_conv_layers):
        residual = outputs
        outputs = norm_fn(outputs, scope="layer_norm_%d" % i, reuse=reuse)
        if (i) % 2 == 0:
            outputs = mx.gluon.nn.Dropout(1.0 - dropout)(outputs)
        outputs = depthwise_separable_convolution(outputs,
                                                  kernel_size=(kernel_size, 1), num_filters=num_filters,
                                                  scope="depthwise_conv_layers_%d" % i, is_training=is_training,
                                                  reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1

    return mx.ndarray.squeeze(outputs, 2), 1

def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.0, sublayers = (1, 1)):
    l, L = sublayers
    # Self attention
    outputs = norm_fn(inputs, scope="layer_norm_1", reuse=reuse)
    outputs = mx.gluon.nn.Dropout(1.0 - dropout)(outputs)
    outputs = multihead_attention(outputs, num_filters,
                                  num_heads=num_heads, seq_len=seq_len, reuse=reuse,
                                  mask=mask, is_training=is_training, bias=bias, dropout=dropout)
    residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
    l += 1
    # Feed-forward
    outputs = norm_fn(residual, scope="layer_norm_2", reuse=reuse)
    outputs = mx.gluon.nn.Dropout(1.0 - dropout)(outputs)
    outputs = conv(outputs, num_filters, True, "relu", name="FFN_1", reuse=reuse)
    outputs = conv(outputs, num_filters, True, None, name="FFN_2", reuse=reuse)
    outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
    l += 1
    return outputs, l

def multihead_attention(queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.0):
    # Self attention
    if memory is None:
        memory = queries

    memory = conv(memory, 2 * units, name="memory_projection", reuse=reuse)
    query = conv(queries, units, name="query_projection", reuse=reuse)
    Q = split_last_dimension(query, num_heads)
    K, V = [split_last_dimension(tensor, num_heads) for tensor in mx.ndarray.split(memory, 2, axis=2)]

    key_depth_per_head = units // num_heads
    Q *= key_depth_per_head ** -0.5
    x = dot_product_attention(Q, K, V,
                              bias=bias,
                              seq_len=seq_len,
                              mask=mask,
                              is_training=is_training,
                              scope="dot_product_attention",
                              reuse=reuse, dropout=dropout)
    return combine_last_two_dimensions(mx.ndarray.transpose().transpose(x, [0, 2, 1, 3]))

def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    shapes = inputs.shape.as_list()
    if len(shapes) > 4:
        raise NotImplementedError
    elif len(shapes) == 4:
        filter_shape = [1, kernel_size, shapes[-1], output_size]
        bias_shape = [1, 1, 1, output_size]
        strides = [1, 1, 1, 1]
    else:
        filter_shape = [kernel_size, shapes[-1], output_size]
        bias_shape = [1, 1, output_size]
        strides = 1

    conv_func = mx.gluon.nn.Conv1D if len(shapes) == 3 else mx.gluon.nn.Conv2D
    use_bias = True
    if bias is None:
        use_bias = False

    outputs = conv_func(output_size, kernel_size=kernel_size, strides=strides, use_bias=use_bias)(inputs)

    if activation is not None:
        if activation == "sigmoid":
            return mx.ndarray.sigmoid(outputs)
        if activation == "relu":
            return mx.ndarray.relu(outputs)
    else:
        return outputs

def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = mx.ndarray.cast(mask, 'float32')
    return inputs + mask_value * (1 - mask)

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None):
    outputs = Separable_Conv(inputs, num_filters, num_filters, withRelu = True)
    return outputs

def mxConv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=False, withBn=True, bn_mom=0.9, workspace=256):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name='%s%s_conv2d' % (name, suffix), workspace=workspace)
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv

def Separable_Conv(data, num_in_channel, num_out_channel, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None, suffix='', depth_mult=1, withBn=True, bn_mom=0.9, workspace=256):
    # original version of Separable Convolution
    # depthwise convolution
    channels       = mx.sym.split(data=data, axis=1, num_outputs=num_in_channel) # for new version of mxnet > 0.8
    depthwise_outs = [mx.sym.Convolution(data=channels[i], num_filter=depth_mult, kernel=kernel,
                           stride=stride, pad=pad, name=name+'_depthwise_kernel_'+str(i), workspace=workspace)
                           for i in range(num_in_channel)]
    depthwise_out = mx.sym.Concat(*depthwise_outs)
    # pointwise convolution
    pointwise_out = mxConv(data=depthwise_out, num_filter=num_out_channel, name=name+'_pointwise_kernel', withBn=False, bn_mom=0.9, workspace=256)
    if withBn:
        pointwise_out = mx.sym.BatchNorm(data=pointwise_out, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    return pointwise_out

def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """

    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = mx.ndarray.reshape(x, mx.ndarray.concat([x.shape(x)[:-1], [n, -1]], 0))

    ret.set_shape(new_shape)
    return mx.ndarray.transpose(ret,[0,2,1,3])

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
    logits = mx.ndarray.transpose(mx.ndarray.multiply(q, k))
    if bias:
        b = mx.ndarray.zeros(logits.shape[-1])
        logits += b
    if mask is not None:
        shapes = [x if x != None else -1 for x in logits.shape.as_list()]
        mask = mx.ndarray.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
        logits = mask_logits(logits, mask)
    weights = mx.ndarray.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = mx.gluon.nn.Dropout(1.0 - dropout)(weights)
    return mx.ndarray.multiply(weights, v)

def combine_last_two_dimensions(x):
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = mx.ndarray.reshape(x, mx.ndarray.concat([x.shape[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.shape[1]
    channels = x.shape[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = mx.nd.arange(length).asnumpy()
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1))
    inv_timescales = min_timescale * mx.ndarray.exp(mx.nd.arange(num_timescales) * -log_timescale_increment)
    scaled_time = mx.ndarray.expand_dims(position, 1) * mx.ndarray.expand_dims(inv_timescales, 0)
    signal = mx.ndarray.concat([mx.ndarray.sin(scaled_time), mx.ndarray.cos(scaled_time)], axis=1)
    signal = mx.ndarray.pad(signal, [[0, 0], [0, mx.ndarray.modulo(channels, 2)]])
    signal = mx.ndarray.reshape(signal, [1, length, channels])
    return signal

def ndim(x):
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
    scope='efficient_trilinear',
    bias_initializer=mx.initializer.Zero(),
    kernel_initializer=mx.initializer.Xavier()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [mx.gluon.nn.Dropout(input_keep_prob)(arg) for arg in args]
    weights4arg0 = mx.gluon.Parameter(
        "linear_kernel4arg0", shape=[arg_size, 1],
        dtype=dtype,
        init=kernel_initializer)
    weights4arg1 = mx.gluon.Parameter(
        "linear_kernel4arg1", shape=[arg_size, 1],
        dtype=dtype,
        init=kernel_initializer)
    weights4mlu = mx.gluon.Parameter(
        "linear_kernel4mul", [1, 1, arg_size],
        dtype=dtype,
        init=kernel_initializer)
    biases = mx.gluon.Parameter(
        "linear_bias", [1],
        dtype=dtype,
        init=bias_initializer)

    subres0 = mx.ndarray.tile(mx.ndarray.dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
    subres1 = mx.ndarray.tile(mx.ndarray.transpose(mx.ndarray.dot(droped_args[1], weights4arg1), perm=(0, 2, 1)),
                      [1, c_maxlen, 1])
    subres2 = mx.ndarray.batch_dot(droped_args[0] * weights4mlu, mx.ndarray.transpose(droped_args[1], perm=(0, 2, 1)))
    res = subres0 + subres1 + subres2
    res = mx.ndarray.add(res, biases)
    return res

def trilinear(args,
            output_size = 1,
            bias = True,
            squeeze=False,
            wd=0.0,
            input_keep_prob= 1.0,
            scope = "trilinear"):
    flat_args = [flatten(arg, 1) for arg in args]
    flat_args = [mx.gluon.nn.Dropout(input_keep_prob)(arg) for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, scope=scope)
    out = reconstruct(flat_out, args[0], 1)
    return mx.ndarray.squeeze(out, -1)

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tensor.shape[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tensor.shape[i] for i in range(start, len(fixed_shape))]
    flat = mx.ndarray.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or ref.shape[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tensor.shape[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = mx.ndarray.reshape(tensor, target_shape)
    return out

def _linear(args,
            output_size,
            bias,
            bias_initializer=mx.initializer.Zero(),
            scope = None,
            kernel_initializer=mx.initializer.Xavier(),
            reuse = None):

  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.shape for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  weights = mx.gluon.Parameter(
      "linear_kernel", shape=[total_arg_size, output_size],
      dtype=dtype,
      init=kernel_initializer)
  if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
  else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
  if not bias:
      return res
  biases = mx.gluon.Parameter(
      "linear_bias", shape=[output_size],
      dtype=dtype,
      init=bias_initializer)

  return nn_ops.bias_add(res, biases)

def total_params():
    total_parameters = 0

    print("Total number of trainable parameters: {}".format(total_parameters))
