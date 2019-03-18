# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:04:10 2018

@author: Administrator
"""

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import activations
#import numpy as np


class MyLayer_one(Layer):# tanh(wxt + b)
    
    def __init__(self,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        
        super(MyLayer_one, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
    def build(self,input_shape):
#        assert len(input_shape1) >= 2
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[0][2],input_shape[1][1]),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[0][1],1),
                                    initializer=self.bias_initializer,
                                    trainable=True)
        self.built = True
#        super(MyLayer_one, self).build(input_shape1,input_shape2)  # Be sure to call this somewhere!

    def call(self, inputs):
        output = K.dot(inputs[0], self.kernel)
        output1 = K.batch_dot(output,inputs[1])
        output2 = K.bias_add(output1, self.bias)
        output2 = self.activation(output2)
        return  output2

    def compute_output_shape(self,input_shape):
        print(input_shape)
        output_shape = list(input_shape[0])
        output_shape[-1] = input_shape[1][-1]
        return tuple(output_shape)