# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras import layers

def SKConv(M=2, r=16, L=32, G=32, name='skconv'):

  def wrapper(inputs):
    inputs_shape = tf.shape(inputs)
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    filters = inputs.get_shape().as_list()[-1]
    d = max(filters//r, L)

    x = inputs

    xs = []
    for m in range(1, M+1):
      if G == 1:
        _x = layers.Conv2D(filters, 3, dilation_rate=m, padding='same',
                          use_bias=False, name=name+'_conv%d'%m)(x)
      else:
        c = filters // G
        _x = layers.DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same',
                                   use_bias=False, name=name+'_conv%d'%m)(x)

        _x = layers.Reshape([h, w, G, c, c], name=name+'_conv%d_reshape1'%m)(_x)
        _x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1),
                          output_shape=[b, h, w, G, c],
                          name=name+'_conv%d_sum'%m)(_x)
        _x = layers.Reshape([h, w, filters],
                           name=name+'_conv%d_reshape2'%m)(_x)


      _x = layers.BatchNormalization(name=name+'_conv%d_bn'%m)(_x)
      _x = layers.Activation('relu', name=name+'_conv%d_relu'%m)(_x)

      xs.append(_x)

    U = layers.Add(name=name+'_add')(xs)
    s = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1,2], keepdims=True),
                      output_shape=[b, 1, 1, filters],
                      name=name+'_gap')(U)

    z = layers.Conv2D(d, 1, name=name+'_fc_z')(s)
    z = layers.BatchNormalization(name=name+'_fc_z_bn')(z)
    z = layers.Activation('relu', name=name+'_fc_z_relu')(z)

    x = layers.Conv2D(filters*M, 1, name=name+'_fc_x')(z)
    x = layers.Reshape([1, 1, filters, M],name=name+'_reshape')(x)
    scale = layers.Softmax(name=name+'_softmax')(x)

    x = layers.Lambda(lambda x: tf.stack(x, axis=-1),
                      output_shape=[b, h, w, filters, M],
                      name=name+'_stack')(xs) # b, h, w, c, M
    x = Axpby(name=name+'_axpby')([scale, x])

    return x
  return wrapper

class Axpby(layers.Layer):
  """
  Do this:
    F = a * X + b * Y + ...
    Shape info:
      a:  B x 1 x 1 x C
      X:  B x H x W x C
      b:  B x 1 x 1 x C
      Y:  B x H x W x C
      ...
      F:  B x H x W x C
  """
  def __init__(self, **kwargs):
        super(Axpby, self).__init__(**kwargs)

  def build(self, input_shape):
        super(Axpby, self).build(input_shape)  # Be sure to call this at the end

  def call(self, inputs):
    """ scale: [B, 1, 1, C, M]
        x: [B, H, W, C, M]
    """
    scale, x = inputs
    f = tf.multiply(scale, x, name='product')
    f = tf.reduce_sum(f, axis=-1, name='sum')
    return f

  def compute_output_shape(self, input_shape):
    return input_shape[0:4]


if __name__ == '__main__':
  from tensorflow.keras.layers import Input
  from tensorflow.keras.models import Model

  inputs = Input([None, None, 32])
  x = SKConv(3, G=1)(inputs)

  m = Model(inputs, x)
  m.summary()

  import numpy as np

  X = np.random.random([2, 224, 224, 32]).astype(np.float32)
  y = m.predict(X)
