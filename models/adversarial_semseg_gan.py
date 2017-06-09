from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, LeakyReLU, Dropout, Flatten, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D
# from keras.regularizers import l2


def build_gan(generator, discriminator, img_shape=(1, 28, 28), dropout_rate=0.25, l2_reg=0.):
    gan_input = Input(shape=img_shape)
    H_gen = generator(gan_input)
    gan_V = discriminator([H_gen, gan_input])
    model = Model(input=[gan_input], output=[H_gen, gan_V], name='dcgan')

    return model