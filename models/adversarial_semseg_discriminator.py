from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, LeakyReLU, Dropout, Flatten, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D,MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
# from keras.regularizers import l2

def build_discriminator(img_shape_gen=(256,256,20), img_shape_orig=(256,256,3), merge_model="standford", discr_model="standford", dropout_rate=0.25, l2_reg=0.):
    generator_output = Input(shape=img_shape_gen)
    orig_img_input = Input(shape=img_shape_orig)

    # Build Discriminative model ...
    H_merged = merge_input(generator_output, orig_img_input, merge_model)
    print(H_merged.shape)
    d_output = discriminator(H_merged, discr_model)

    model = Model(input=[generator_output, orig_img_input], output=d_output)
    return model

def merge_input(generator_output, orig_img_input, merge_model="standford"):
    if merge_model=="standford":
        return merge_standford(generator_output, orig_img_input)
    elif merge_model=="basic":
        return merge_basic(generator_output, orig_img_input)
    elif merge_model=="product":
        return merge_product(generator_output, orig_img_input)
    elif merge_model=="scaling":
        return merge_scaling(generator_output, orig_img_input)
    else:
        raise ValueError('Unknown merge model name for GANs')

def discriminator(H_merged, discr_model="standford"):
    if discr_model=="standford":
        return discriminator_standford(H_merged)
    elif discr_model=="largeFOV_light":
        return discriminator_LargeFOV_Light(H_merged)
    elif discr_model=="largeFOV":
        return discriminator_LargeFOV(H_merged)
    elif discr_model=="smallFOV_light":
        return discriminator_SmallFOV_Light(H_merged)
    elif discr_model=="smallFOV":
        return discriminator_SmallFOV(H_merged)
    else:
        raise ValueError('Unknown discriminator model name for GANs')

def merge_standford(generator_output, orig_img_input):
    H_gen = Convolution2D(64, 3, 3, border_mode='same',
                      activation='relu', name="generator_input")(generator_output)

    H_orig = Convolution2D(16, 5, 5, border_mode='same',
                      activation='relu', name="orig_img_input")(orig_img_input)

    #H_orig = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(H_orig)

    H_orig = Convolution2D(64, 5, 5, border_mode='same',
                      activation='relu')(H_orig)

    #H_orig = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(H_orig)

    H_merged = merge([H_gen, H_orig], mode='concat', concat_axis=3)
    return H_merged

def merge_basic(generator_output, img_shape_orig):
    return generator_output

def merge_product(generator_output, orig_img_input):
    return ValueError("Merge protuct is not fully implemented")
    #error related to None dimension on layers and posterior flatten+dense on discriminator
    generator_output = K.permute_dimensions(generator_output, (3,0,1,2))
    orig_img_input = K.permute_dimensions(orig_img_input, (3,0,1,2))

    H = []
    for i in range(orig_img_input.shape[0]):
        for j in range(generator_output.shape[0]):
            new_prod = tf.multiply(orig_img_input[i], generator_output[j])
            #new_prod = merge([orig_img_input[i], generator_output[j]], mode='mul')
            H.append(new_prod)


    H_merged = K.stack(H)
    H_merged = K.permute_dimensions(H_merged, (1,2,3,0))
    return H_merged

def merge_scaling(generator_output, orig_img_input):
    #TODO
    return ValueError("Merge scaling is not yet implemented")


def discriminator_standford(H_merged):
    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)

    H_merged = Convolution2D(256, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)

    H_merged = Convolution2D(512, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    #H_merged = Convolution2D(2, 3, 3, border_mode='same')(H_merged)
    #d_output = H_merged
    
    #mean = K.mean(H_merged, axis=1)
    #mean = K.mean(mean, axis=1)

    d_output = Flatten()(H_merged)
    d_output = Dense(2, activation='softmax')(d_output)

    return d_output

def discriminator_LargeFOV_Light(H_merged):
    H_merged = Convolution2D(96, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)


    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)

    H_merged = Convolution2D(256, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    #H_merged = Convolution2D(2, 3, 3, border_mode='same')(H_merged)
    #d_output = H_merged
    
    H_merged = Flatten()(H_merged)
    d_output = Dense(2, activation='softmax')(H_merged)

    return d_output

def discriminator_LargeFOV(H_merged):
    H_merged = Convolution2D(96, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)


    H_merged = Convolution2D(256, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(256, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)

    H_merged = Convolution2D(512, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    #H_merged = Convolution2D(2, 3, 3, border_mode='same')(H_merged)
    #d_output = H_merged

    H_merged = Flatten()(H_merged)
    d_output = Dense(2, activation='softmax')(H_merged)

    return d_output

def discriminator_SmallFOV_Light(H_merged):
    H_merged = Convolution2D(96, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 1, 1, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)


    H_merged = Convolution2D(128, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 1, 1, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)

    H_merged = Convolution2D(256, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    #H_merged = Convolution2D(2, 1, 1, border_mode='same')(H_merged)
    #d_output = H_merged

    H_merged = Flatten()(H_merged)
    d_output = Dense(2, activation='softmax')(H_merged)

    return d_output

def discriminator_SmallFOV(H_merged):
    H_merged = Convolution2D(96, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(128, 1, 1, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)


    H_merged = Convolution2D(256, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = Convolution2D(256, 1, 1, border_mode='same',
                      activation='relu')(H_merged)

    H_merged = MaxPooling2D(pool_size=(2, 2))(H_merged)

    H_merged = Convolution2D(512, 3, 3, border_mode='same',
                      activation='relu')(H_merged)

    #H_merged = Convolution2D(2, 1, 1, border_mode='same')(H_merged)
    #d_output = H_merged

    H_merged = Flatten()(H_merged)
    d_output = Dense(2, activation='softmax')(H_merged)

    return d_output
