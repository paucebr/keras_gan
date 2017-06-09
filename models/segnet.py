# Keras imports
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D, UpSampling2D, Deconvolution2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.regularizers import l2

# Custom layers import
from ourlayers import (CropLayer2D, NdSoftmax, DePool2D)

from keras import backend as K


def channel_idx():
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        return 1
    else:
        return 3


# Downsample blocks of the basic-segnet
def downsampling_block_basic(inputs, n_filters, filter_size, 
                             init='glorot_uniform', W_regularizer=None):
    # This extra padding is used to prevent problems with different input
    # sizes. At the end the crop layer remove extra paddings
    pad = ZeroPadding2D(padding=(1, 1))(inputs)
    conv = Convolution2D(n_filters, filter_size, filter_size, init=init, 
                         border_mode='same', W_regularizer=W_regularizer)(pad)
    """
    IMPORTANT bn_mode = 2
    """                     
    bn_mode = 2    
    bn = BatchNormalization(mode=bn_mode, axis=channel_idx())(conv)
    act = Activation('relu')(conv)
    maxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)
    return maxp


# Upsample blocks of the basic-segnet
def upsampling_block_basic(inputs, n_filters, filter_size, unpool_layer=None,
                           init='glorot_uniform', W_regularizer=None, use_unpool=True):
    if use_unpool:
        up = DePool2D(unpool_layer)(inputs)
    else:
        up = UpSampling2D()(inputs)
        
    conv = Convolution2D(n_filters, filter_size, filter_size, init=init, 
                         border_mode='same', W_regularizer=W_regularizer)(up)
    """
    IMPORTANT bn_mode = 2
    """                     
    bn_mode = 2    
    bn = BatchNormalization(mode=bn_mode, axis=channel_idx())(conv)
    return conv


# Create model of basic segnet
def build_segnet_basic(inputs, n_classes, depths=[64, 64, 64, 64],
                       filter_size=7, l2_reg=0.):
    """ encoding layers """
    enc1 = downsampling_block_basic(inputs, depths[0], filter_size, W_regularizer=l2(l2_reg))
    enc2 = downsampling_block_basic(enc1, depths[1], filter_size, W_regularizer=l2(l2_reg))
    enc3 = downsampling_block_basic(enc2, depths[2], filter_size, W_regularizer=l2(l2_reg))
    enc4 = downsampling_block_basic(enc3, depths[3], filter_size, W_regularizer=l2(l2_reg))

    """ decoding layers """
    dec1 = upsampling_block_basic(enc4, depths[3], filter_size, enc4, W_regularizer=l2(l2_reg))
    dec2 = upsampling_block_basic(dec1, depths[2], filter_size, enc3, W_regularizer=l2(l2_reg))
    dec3 = upsampling_block_basic(dec2, depths[1], filter_size, enc2, W_regularizer=l2(l2_reg))
    dec4 = upsampling_block_basic(dec3, depths[0], filter_size, enc1, W_regularizer=l2(l2_reg))

    """ logits """
    l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec4)
    score = CropLayer2D(inputs, name='score')(l1)
    softmax_segnet = NdSoftmax()(score)

    # Complete model
    model = Model(input=inputs, output=softmax_segnet)

    return model


# Downsampling block of the VGG
def downsampling_block_vgg(inputs, n_conv, n_filters, filter_size, layer_id,
                           l2_reg=None, activation='relu',
                           init='glorot_uniform', border_mode='same'):
    #conv = ZeroPadding2D(padding=(1, 1))(inputs)
    conv = inputs
    for i in range(1, n_conv+1):
        name = 'conv' + str(layer_id) + '_' + str(i)
        conv = Convolution2D(n_filters, filter_size, filter_size, init,
                             border_mode=border_mode, name=name,
                             W_regularizer=l2(l2_reg))(conv)
        """
        IMPORTANT bn_mode = 2
        """                     
        bn_mode = 2    
        conv = BatchNormalization(mode=bn_mode, axis=channel_idx(),
                                  name=name + '_bn')(conv)
        conv = Activation(activation, name=name + '_relu')(conv)
    conv = MaxPooling2D((2, 2), (2, 2), name='pool'+str(layer_id))(conv)
    return conv



# Upsampling block of the VGG
def upsampling_block_vgg(inputs, n_conv, n_filters, filter_size, layer_id,
                         l2_reg=None, unpool_layer=None, activation='relu',
                         init='glorot_uniform', border_mode='same',
                         use_unpool=False):
    if use_unpool:
        conv = DePool2D(unpool_layer, name='upsample'+str(layer_id))(inputs)
    else:
        conv = UpSampling2D()(inputs)


    for i in range(n_conv+1, 1, -1):
        conv = Convolution2D(n_filters, filter_size, filter_size, init,
                             border_mode=border_mode, 
                             name='conv'+str(layer_id)+'_'+str(i)+'_D',
                             W_regularizer=l2(l2_reg))(conv)
        """
        IMPORTANT bn_mode = 2
        """                     
        bn_mode = 2    
        conv = BatchNormalization(mode=bn_mode, axis=channel_idx())(conv)
        conv = Activation(activation)(conv)
    return conv


# Create model of VGG Segnet
def build_segnet_vgg(inputs, n_classes, l2_reg=0.):
    """ encoding layers """
    enc1 = downsampling_block_vgg(inputs, 2, 64, 3, 1, l2_reg)
    enc2 = downsampling_block_vgg(enc1, 2, 128, 3, 2, l2_reg)
    enc3 = downsampling_block_vgg(enc2, 3, 256, 3, 3, l2_reg)
    enc4 = downsampling_block_vgg(enc3, 3, 512, 3, 4, l2_reg)
    enc5 = downsampling_block_vgg(enc4, 3, 512, 3, 5, l2_reg)

    """ decoding layers """
    dec5 = upsampling_block_vgg(enc5, 3, 512, 3, 5, l2_reg, enc5)
    dec4 = upsampling_block_vgg(dec5, 3, 512, 3, 4, l2_reg, enc4)
    #dec4 = upsampling_block_vgg(enc4, 3, 512, 3, 4, l2_reg, enc4)
    dec3 = upsampling_block_vgg(dec4, 3, 256, 3, 3, l2_reg, enc3)
    dec2 = upsampling_block_vgg(dec3, 2, 128, 3, 2, l2_reg, enc2)
    dec1 = upsampling_block_vgg(dec2, 2, 64, 3, 1, l2_reg, enc1)

    """ logits """
    #dec1 = inputs   
    l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec1)
    #score = CropLayer2D(inputs, name='score')(l1)
    # No es pot fer aixo:
    # score = Activation('softmax')(crop)
    # perque
    # ValueError: Cannot apply softmax to a tensor that is not 2D or 3D. Here, ndim=4
    softmax_segnet = NdSoftmax()(l1)

    # Complete model
    model = Model(input=inputs, output=softmax_segnet)

    return model





def build_segnet(img_shape=(3, None, None), n_classes=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 freeze_layers_from=None, use_unpool=False, basic=False):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Input layer
    inputs = Input(img_shape,name='input_segnet')

    # Create basic Segnet
    if basic:
        model = build_segnet_basic(inputs, n_classes, depths=[64, 64, 64, 64],
                                   filter_size=3, l2_reg=l2_reg)
    else:
        model = build_segnet_vgg(inputs, n_classes, l2_reg)

 
    return model



if __name__ == '__main__':
    print ('BUILD full segnet')
    model_full = build_segnet(img_shape=(256, 256, 3), n_classes=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 freeze_layers_from=None, use_unpool=False, basic=False)
    print ('COMPILING full segnet')
    model_full.compile(loss="binary_crossentropy", optimizer="rmsprop")
    model_full.summary()
    print ('END COMPILING full segnet')

    #print('')
    #print ('BUILD basic segnet')
    #model_basic = build_segnet(img_shape=(3, 360, 480), n_classes=8, l2_reg=0.,
    #             init='glorot_uniform', path_weights=None,
    #             freeze_layers_from=None, use_unpool=False, basic=True)
    #print ('COMPILING basic segnet')
    #model_basic.compile(loss="binary_crossentropy", optimizer="rmsprop")
    #model_basic.summary()
    #print ('END COMPILING basic segnet')
