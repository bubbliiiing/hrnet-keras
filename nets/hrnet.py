import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          DepthwiseConv2D, Dropout, GlobalAveragePooling2D,
                          Input, Lambda, Softmax, ZeroPadding2D)
from keras.models import Model

from nets.backbone import HRnet_Backbone, UpsampleLike


def HRnet(input_shape, num_classes=21, backbone="hrnetv2_w18"):
    inputs = Input(shape=input_shape)
    x, num_filters = HRnet_Backbone(inputs, backbone)

    x0_0 = x[0]
    x0_1 = UpsampleLike()([x[1], x[0]])
    x0_2 = UpsampleLike()([x[2], x[0]])
    x0_3 = UpsampleLike()([x[3], x[0]])

    x = Concatenate(axis=-1)([x0_0, x0_1, x0_2, x0_3])

    x = Conv2D(np.sum(num_filters), 1, strides=(1, 1))(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation("relu")(x)
    x = Conv2D(num_classes, 1, strides=(1, 1))(x)

    shape = tf.keras.backend.int_shape(inputs)
    x = Lambda(lambda xx : tf.image.resize_images(xx, shape[1:3], align_corners=True))(x)
    x = Softmax()(x)
    model = Model(inputs, x, name="HRnet")
    return model
