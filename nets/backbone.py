import keras
from keras.layers.convolutional import ZeroPadding2D
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D, Input, Layer,
                          UpSampling2D, add)
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model


class UpsampleLike(Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.image.resize_images(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False, name=""):
    x = Conv2D(out_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', name=name+'.conv1')(input)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn1')(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', name=name+'.conv2')(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn2')(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal', name=name+'.downsample.0')(input)
        residual = BatchNormalization(epsilon=1e-5, name=name+'.downsample.1')(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False, name=""):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal', name=name+'.conv1')(input)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn1')(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', name=name+'.conv2')(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn2')(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal', name=name+'.conv3')(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'.bn3')(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal', name=name+'.downsample.0')(input)
        residual = BatchNormalization(epsilon=1e-5, name=name+'.downsample.1')(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x

def stem_net(input):
    x = ZeroPadding2D(((1, 1),(1, 1)))(input)
    x = Conv2D(64, 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="conv1")(x)
    x = BatchNormalization(epsilon=1e-5, name="bn1")(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(((1, 1),(1, 1)))(x)
    x = Conv2D(64, 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="conv2")(x)
    x = BatchNormalization(epsilon=1e-5, name="bn2")(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 256, with_conv_shortcut=True, name="layer1.0")
    x = bottleneck_Block(x, 256, with_conv_shortcut=False, name="layer1.1")
    x = bottleneck_Block(x, 256, with_conv_shortcut=False, name="layer1.2")
    x = bottleneck_Block(x, 256, with_conv_shortcut=False, name="layer1.3")
    return x

def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal', name="transition1.0.0")(x)
    x0 = BatchNormalization(epsilon=1e-5, name="transition1.0.1")(x0)
    x0 = Activation('relu')(x0)

    x1 = ZeroPadding2D(((1, 1),(1, 1)))(x)
    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name = "transition1.1.0.0")(x1)
    x1 = BatchNormalization(epsilon=1e-5, name="transition1.1.0.1")(x1)
    x1 = Activation('relu')(x1)
    return [x0, x1]

def make_stage2(x_list, out_filters_list=[32, 64]):
    x0, x1 = x_list

    x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage2.0.branches.0.0")
    x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage2.0.branches.0.1")
    x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage2.0.branches.0.2")
    x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage2.0.branches.0.3")

    x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage2.0.branches.1.0")
    x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage2.0.branches.1.1")
    x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage2.0.branches.1.2")
    x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage2.0.branches.1.3")

    x0_0 = x0
    x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal', name="stage2.0.fuse_layers.0.1.0")(x1)
    x0_1 = BatchNormalization(epsilon=1e-5, name="stage2.0.fuse_layers.0.1.1")(x0_1)
    x0_1 = UpsampleLike(name="Upsample1")([x0_1, x0_0])
    x0_out = add([x0_0, x0_1])
    x0_out = Activation('relu')(x0_out)

    x1_0 = ZeroPadding2D(((1, 1),(1, 1)))(x0)
    x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage2.0.fuse_layers.1.0.0.0")(x1_0)
    x1_0 = BatchNormalization(epsilon=1e-5, name="stage2.0.fuse_layers.1.0.0.1")(x1_0)
    x1_1 = x1
    x1_out = add([x1_0, x1_1])
    x1_out = Activation('relu')(x1_out)
    
    return x0_out, x1_out

def transition_layer2(x, out_filters_list=[32, 64, 128]):
    x2 = ZeroPadding2D(((1, 1),(1, 1)))(x[1])
    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="transition2.2.0.0")(x2)
    x2 = BatchNormalization(epsilon=1e-5, name="transition2.2.0.1")(x2)
    x2 = Activation('relu')(x2)
    return [x[0], x[1], x2]

def make_stage3(x_list, num_modules, out_filters_list=[32, 64, 128]):
    for i in range(num_modules):
        x0, x1, x2 = x_list
        
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.0.0")
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.0.1")
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.0.2")
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.0.3")

        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.1.0")
        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.1.1")
        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.1.2")
        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.1.3")

        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.2.0")
        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.2.1")
        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.2.2")
        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage3." + str(i) + ".branches.2.3")

        x0_0 = x0
        x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.0.1.0")(x1)
        x0_1 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.0.1.1")(x0_1)
        x0_1 = UpsampleLike(name="Upsample." + str(i) + ".2")([x0_1, x0_0])
        x0_2 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.0.2.0")(x2)
        x0_2 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.0.2.1")(x0_2)
        x0_2 = UpsampleLike(name="Upsample." + str(i) + ".3")([x0_2, x0_0])
        x0_out = add([x0_0, x0_1, x0_2])
        x0_out = Activation('relu')(x0_out)

        
        x1_0 = ZeroPadding2D(((1, 1),(1, 1)))(x0)
        x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.1.0.0.0")(x1_0)
        x1_0 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.1.0.0.1")(x1_0)
        x1_1 = x1
        x1_2 = Conv2D(out_filters_list[1], 1, use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.1.2.0")(x2)
        x1_2 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.1.2.1")(x1_2)
        x1_2 = UpsampleLike(name="Upsample." + str(i) + ".4")([x1_2, x1_1])
        x1_out = add([x1_0, x1_1, x1_2])
        x1_out = Activation('relu')(x1_out)

        x2_0 = ZeroPadding2D(((1, 1),(1, 1)))(x0)
        x2_0 = Conv2D(out_filters_list[0], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.2.0.0.0")(x2_0)
        x2_0 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.2.0.0.1")(x2_0)
        x2_0 = Activation('relu')(x2_0)
        x2_0 = ZeroPadding2D(((1, 1),(1, 1)))(x2_0)
        x2_0 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.2.0.1.0")(x2_0)
        x2_0 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.2.0.1.1")(x2_0)
        x2_1 = ZeroPadding2D(((1, 1),(1, 1)))(x1)
        x2_1 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage3." + str(i) + ".fuse_layers.2.1.0.0")(x2_1)
        x2_1 = BatchNormalization(epsilon=1e-5, name="stage3." + str(i) + ".fuse_layers.2.1.0.1")(x2_1)
        x2_2 = x2
        x2_out = add([x2_0, x2_1, x2_2])
        x2_out = Activation('relu')(x2_out)
        
        x_list = [x0_out, x1_out, x2_out]

    return x_list

def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x3 = ZeroPadding2D(((1, 1),(1, 1)))(x[2])
    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="transition3.3.0.0")(x3)
    x3 = BatchNormalization(epsilon=1e-5, name="transition3.3.0.1")(x3)
    x3 = Activation('relu')(x3)

    return [x[0], x[1], x[2], x3]

def make_stage4(x_list, num_modules, out_filters_list=[32, 64, 128, 256]):
    for i in range(num_modules):
        x0, x1, x2, x3 = x_list
        
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.0.0")
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.0.1")
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.0.2")
        x0 = basic_Block(x0, out_filters_list[0], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.0.3")

        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.1.0")
        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.1.1")
        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.1.2")
        x1 = basic_Block(x1, out_filters_list[1], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.1.3")

        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.2.0")
        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.2.1")
        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.2.2")
        x2 = basic_Block(x2, out_filters_list[2], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.2.3")

        x3 = basic_Block(x3, out_filters_list[3], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.3.0")
        x3 = basic_Block(x3, out_filters_list[3], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.3.1")
        x3 = basic_Block(x3, out_filters_list[3], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.3.2")
        x3 = basic_Block(x3, out_filters_list[3], with_conv_shortcut=False, name="stage4." + str(i) + ".branches.3.3")

        x0_0 = x0
        x0_1 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.0.1.0")(x1)
        x0_1 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.0.1.1")(x0_1)
        x0_1 = UpsampleLike(name="Upsample." + str(i) + ".5")([x0_1, x0_0])
        x0_2 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.0.2.0")(x2)
        x0_2 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.0.2.1")(x0_2)
        x0_2 = UpsampleLike(name="Upsample." + str(i) + ".6")([x0_2, x0_0])
        x0_3 = Conv2D(out_filters_list[0], 1, use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.0.3.0")(x3)
        x0_3 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.0.3.1")(x0_3)
        x0_3 = UpsampleLike(name="Upsample." + str(i) + ".7")([x0_3, x0_0])
        x0_out = add([x0_0, x0_1, x0_2, x0_3])
        x0_out = Activation('relu')(x0_out)

        x1_0 = ZeroPadding2D(((1, 1),(1, 1)))(x0)
        x1_0 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.1.0.0.0")(x1_0)
        x1_0 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.1.0.0.1")(x1_0)
        x1_1 = x1
        x1_2 = Conv2D(out_filters_list[1], 1, use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.1.2.0")(x2)
        x1_2 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.1.2.1")(x1_2)
        x1_2 = UpsampleLike(name="Upsample." + str(i) + ".8")([x1_2, x1_1])
        x1_3 = Conv2D(out_filters_list[1], 1, use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.1.3.0")(x3)
        x1_3 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.1.3.1")(x1_3)
        x1_3 = UpsampleLike(name="Upsample." + str(i) + ".9")([x1_3, x1_1])
        x1_out = add([x1_0, x1_1, x1_2, x1_3])
        x1_out = Activation('relu')(x1_out)

        x2_0 = ZeroPadding2D(((1, 1),(1, 1)))(x0)
        x2_0 = Conv2D(out_filters_list[0], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.2.0.0.0")(x2_0)
        x2_0 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.2.0.0.1")(x2_0)
        x2_0 = Activation('relu')(x2_0)
        x2_0 = ZeroPadding2D(((1, 1),(1, 1)))(x2_0)
        x2_0 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.2.0.1.0")(x2_0)
        x2_0 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.2.0.1.1")(x2_0)
        x2_1 = ZeroPadding2D(((1, 1),(1, 1)))(x1)
        x2_1 = Conv2D(out_filters_list[2], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.2.1.0.0")(x2_1)
        x2_1 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.2.1.0.1")(x2_1)
        x2_2 = x2
        x2_3 = Conv2D(out_filters_list[2], 1, use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.2.3.0")(x3)
        x2_3 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.2.3.1")(x2_3)
        x2_3 = UpsampleLike(name="Upsample." + str(i) + ".10")([x2_3, x2_2])
        x2_out = add([x2_0, x2_1, x2_2, x2_3])
        x2_out = Activation('relu')(x2_out)
        
        x3_0 = ZeroPadding2D(((1, 1),(1, 1)))(x0)
        x3_0 = Conv2D(out_filters_list[0], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.3.0.0.0")(x3_0)
        x3_0 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.3.0.0.1")(x3_0)
        x3_0 = Activation('relu')(x3_0)
        x3_0 = ZeroPadding2D(((1, 1),(1, 1)))(x3_0)
        x3_0 = Conv2D(out_filters_list[0], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.3.0.1.0")(x3_0)
        x3_0 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.3.0.1.1")(x3_0)
        x3_0 = Activation('relu')(x3_0)
        x3_0 = ZeroPadding2D(((1, 1),(1, 1)))(x3_0)
        x3_0 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.3.0.2.0")(x3_0)
        x3_0 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.3.0.2.1")(x3_0)
        x3_1 = ZeroPadding2D(((1, 1),(1, 1)))(x1)
        x3_1 = Conv2D(out_filters_list[1], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.3.1.0.0")(x3_1)
        x3_1 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.3.1.0.1")(x3_1)
        x3_1 = Activation('relu')(x3_1)
        x3_1 = ZeroPadding2D(((1, 1),(1, 1)))(x3_1)
        x3_1 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.3.1.1.0")(x3_1)
        x3_1 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.3.1.1.1")(x3_1)
        x3_2 = ZeroPadding2D(((1, 1),(1, 1)))(x2)
        x3_2 = Conv2D(out_filters_list[3], 3, strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name="stage4." + str(i) + ".fuse_layers.3.2.0.0")(x3_2)
        x3_2 = BatchNormalization(epsilon=1e-5, name="stage4." + str(i) + ".fuse_layers.3.2.0.1")(x3_2)
        x3_3 = x3
        x3_out = add([x3_0, x3_1, x3_2, x3_3])
        x3_out = Activation('relu')(x3_out)
        x_list = [x0_out, x1_out, x2_out, x3_out]

    return x_list


def make_head(x_list, head_channels = [128, 256, 512, 1024]):
    x0, x1, x2, x3  = x_list

    x0 = bottleneck_Block(x0, head_channels[0], with_conv_shortcut=True, name = "incre_modules.0.0")

    y = ZeroPadding2D(((1, 1),(1, 1)))(x0)
    y = Conv2D(head_channels[1], 3, strides=(2, 2), padding='valid', use_bias=True, kernel_initializer='he_normal', name="downsamp_modules.0.0")(y)
    y = BatchNormalization(epsilon=1e-5, name="downsamp_modules.0.1")(y)
    y = Activation('relu')(y)
    x1 = bottleneck_Block(x1, head_channels[1], with_conv_shortcut=True, name = "incre_modules.1.0")
    y = add([x1, y])

    y = ZeroPadding2D(((1, 1),(1, 1)))(y)
    y = Conv2D(head_channels[2], 3, strides=(2, 2), padding='valid', use_bias=True, kernel_initializer='he_normal', name="downsamp_modules.1.0")(y)
    y = BatchNormalization(epsilon=1e-5, name="downsamp_modules.1.1")(y)
    y = Activation('relu')(y)
    x2 = bottleneck_Block(x2, head_channels[2], with_conv_shortcut=True, name = "incre_modules.2.0")
    y = add([x2, y])

    y = ZeroPadding2D(((1, 1),(1, 1)))(y)
    y = Conv2D(head_channels[3], 3, strides=(2, 2), padding='valid', use_bias=True, kernel_initializer='he_normal', name="downsamp_modules.2.0")(y)
    y = BatchNormalization(epsilon=1e-5, name="downsamp_modules.2.1")(y)
    y = Activation('relu')(y)
    x3 = bottleneck_Block(x3, head_channels[3], with_conv_shortcut=True, name = "incre_modules.3.0")
    y = add([x3, y])

    y = Conv2D(2048, 1, strides=(1, 1), padding='same', use_bias=True, kernel_initializer='he_normal', name="final_layer.0")(y)
    y = BatchNormalization(epsilon=1e-5, name="final_layer.1")(y)
    y = Activation('relu')(y)
    return y

def HRnet_Backbone(inputs, phi):
    num_filters = {
        'hrnetv2_w18' : [18, 36, 72, 144],
        'hrnetv2_w32' : [32, 64, 128, 256],
        'hrnetv2_w48' : [48, 96, 192, 384],
    }[phi]

    x = stem_net(inputs)

    x = transition_layer1(x, out_filters_list = [num_filters[0], num_filters[1]])
    x = make_stage2(x, out_filters_list = [num_filters[0], num_filters[1]])

    x = transition_layer2(x, out_filters_list = [num_filters[0], num_filters[1], num_filters[2]])
    x = make_stage3(x, 4, out_filters_list = [num_filters[0], num_filters[1], num_filters[2]])

    x = transition_layer3(x, out_filters_list = [num_filters[0], num_filters[1], num_filters[2], num_filters[3]])
    x = make_stage4(x, 3, out_filters_list = [num_filters[0], num_filters[1], num_filters[2], num_filters[3]])

    return x, num_filters
