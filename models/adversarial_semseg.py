import os
import numpy as np
import random
# Keras imports
import keras.models as kmodels
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input, merge, Merge
from keras.utils.visualize_util import plot
from model import Model
from adversarial_semseg_discriminator import build_discriminator
from adversarial_semseg_gan import build_gan
from segnet import build_segnet
from unet import build_unet
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools.two_image_iterator import Two_Image_Iterator
from sklearn.preprocessing import OneHotEncoder
from tools.save_images_adv import save_img3
import math

from tools.adversarial_batches import *
from tools.adversarial_tools import *
from metrics.adversarial_semseg_metrics import *


class Adversarial_Semseg(Model):
    def __init__(self, cf, img_shape):
        self.cf = cf
        self.img_shape = img_shape
        self.n_classes = self.cf.dataset.n_classes

        # Make and compile the generator
        self.g_optimizer = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08) #this should be parametrized
        self.g_img_shape = img_shape
        self.generator = self.make_generator(self.g_img_shape,
                                             self.cf.dataset.n_classes,
                                             self.g_optimizer,
                                             the_loss='categorical_crossentropy',
                                             metrics=[])


        # Make and compile the discriminator
        self.d_optimizer = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08) #this should be parametrized
        self.d_img_shape = img_shape
        self.discriminator = self.make_discriminator(self.generator,
                                                     self.d_img_shape,
                                                     self.d_optimizer,
                                                     the_loss='binary_crossentropy',
                                                     metrics=[])

        # Freeze weights in the discriminator for stacked training
        make_trainable(self.discriminator, False)
        self.dcgan_optimizer = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08) #this should be parametrized
        # Make and compile the GAN
        self.GAN = self.make_gan(self.g_img_shape, self.dcgan_optimizer,
                                 the_loss=['categorical_crossentropy','binary_crossentropy'],
                                 loss_weights=[0.1, 1.0],
                                 metrics=[])
        #self.GAN = self.make_gan(self.g_img_shape, self.dcgan_optimizer,
        #                            the_loss=['categorical_crossentropy'],
        #                            metrics=[])

    # Make generator
    def make_generator(self, img_shape, n_classes, optimizer,
                       the_loss='categorical_crossentropy', metrics=[]):
        # Build model
        generator = build_segnet(img_shape, n_classes=n_classes, l2_reg=0.)

        # Compile model
        generator.compile(loss=the_loss, metrics=metrics, optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('Generator')
            generator.summary()
            plot(generator, to_file=os.path.join(self.cf.savepath, 'model_generator.png'))
        return generator

    # Make discriminator
    def make_discriminator(self, generator, img_shape, optimizer,
                           the_loss='binary_crossentropy', metrics=[]):
        # Build model
        discriminator = build_discriminator(generator.output_shape[1:], img_shape, 
                            self.cf.merge_model, self.cf.discr_model, dropout_rate=0.25, l2_reg=0.)

        # Compile model
        discriminator.compile(loss=the_loss, metrics=metrics, optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('Discriminator')
            discriminator.summary()
            plot(discriminator, to_file=os.path.join(self.cf.savepath, 'model_discriminator.png'))

        return discriminator

    # Make GAN
    def make_gan(self, img_shape, optimizer,
                the_loss=['categorical_crossentropy','binary_crossentropy'], loss_weights=[1., 1.], metrics=[]):
    #             the_loss=['categorical_crossentropy','binary_crossentropy'], loss_weights=[0.1, 1.], metrics=[]):
    #                the_loss=['categorical_crossentropy','binary_crossentropy'], loss_weights=[1.], metrics=[]):
        
        # Build stacked GAN model
        GAN = build_gan(self.generator, self.discriminator, img_shape, dropout_rate=0.25, l2_reg=0.)

        # Compile model
        GAN.compile(loss=the_loss, loss_weights=loss_weights, metrics=metrics, optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('GAN')
            GAN.summary()
            plot(GAN, to_file=os.path.join(self.cf.savepath, 'model_GAN.png'))

        return GAN
     

    #pretrains not yet defined, they could be usefull later, keep in mind
    def train(self, train_gen, valid_gen, cb):
        if (self.cf.train_model):
            #losses = {"d": [], "g": []}
            accuracy = {"train":[], "valid":[], "test":[]}

            for i in range(self.cf.dataset.n_classes):
                accuracy["train"].append([])
                accuracy["valid"].append([])
                accuracy["test"].append([])

            print('\n > Training the model...')
            n_iters_discr = self.cf.n_iters_discr
            # 18 es una epoca perque hi ha 367 imatges i el batch_size es 20
            n_iters_gen = self.cf.n_iters_gen
            # es molt diferent fer 1 que 10 ! perque ?

            dataset_path = self.cf.dataset.path
            train_path = os.path.join(dataset_path, 'train')
            train_it = Two_Image_Iterator(train_path,batch_size=self.cf.batch_size_train, 
                        target_size=self.img_shape[:-1])

            valid_path = os.path.join(dataset_path, 'valid')
            valid_it = Two_Image_Iterator(valid_path,batch_size=self.cf.batch_size_valid, 
                        target_size=self.img_shape[:-1])
            
            loss_discr = []
            loss_gen = []

            while train_it.epochs_completed() < self.cf.n_epochs:
                # train discriminator
                input_img_discr, x_discr, y_discr = get_batch_for_discriminator(train_it, self.cf.dataset.n_classes, 
                                                            self.generator, mode='mix')
                                                                                                                                                                
                make_trainable(self.generator, False) # maybe not necessary, just in case 
                make_trainable(self.discriminator, True)                    
                for i in range(n_iters_discr):                                                                         
                    ld = self.discriminator.train_on_batch([x_discr, input_img_discr], y_discr)
                    loss_discr.append(ld)
               
                # train gan/generator by training gan with discriminator fixed
                make_trainable(self.discriminator, False)                    
                make_trainable(self.generator, True) # maybe not necessary, just in case

                for i in range(n_iters_gen):                                                                         
                    input_image, gt_one_hot, y_gen = get_batch_for_generator(train_it, self.cf.dataset.n_classes)                   
                    lg = self.GAN.train_on_batch(input_image, [gt_one_hot, y_gen])
                    #lg = self.GAN.train_on_batch(input_image, [y_gen])
                    loss_gen.append(lg)


                # display acurracy and loss metrics
                if train_it.total_batches_seen % self.cf.display_every_batches == 0: 
                    #losses["d"].append(ld)
                    #losses["g"].append(lg)

                    mIoU = get_validation_metrics(self.generator, self.cf.dataset.path, self.cf.dataset.n_classes, 
                                self.cf.batch_size_valid, self.img_shape[:-1])


                    for i in range(self.cf.dataset.n_classes):
                        accuracy["train"][i].append(mIoU["train"][i])
                        accuracy["valid"][i].append(mIoU["valid"][i])
                        accuracy["test"][i].append(mIoU["test"][i])

                    print('epoch {}, batch {}, loss discriminator {}, loss gan/generator {}'.\
                    format(train_it.epochs_completed(), train_it.total_batches_seen, ld, lg))
                    plot_loss(np.array(loss_gen), np.array(loss_discr), self.cf.savepath, [0.1, 1.])
                    plot_accuracy(accuracy, self.cf.dataset.classes, self.cf.dataset.n_classes, self.cf.savepath, self.cf.dataset.color_map, 4)
                    
                

                # save semseg validation samples and network weights
                if self.cf.save_results and train_it.total_batches_seen % self.cf.save_every_batches == 0:
                    input_image_save, gt_one_hot_save, y_gen_save = get_batch_for_generator(valid_it, self.cf.dataset.n_classes)  
                    pred_save = self.generator.predict(input_image_save)
                    save_img3(input_image_save*255, gt_one_hot_save, pred_save, self.cf.savepath, valid_it.epochs_completed(),
                      self.cf.dataset.color_map, self.cf.dataset.classes, "valid_"+str(valid_it.epochs_completed()),
                      self.cf.dataset.void_class, 2)
                    
                    self.generator.save_weights(self.cf.savepath+'/test_gen.h5')
                    self.discriminator.save_weights(self.cf.savepath+'/test_disc.h5')
                    self.GAN.save_weights(self.cf.savepath+'/test_gan.h5')
                    np.savez(self.cf.savepath+'/losses.npz',loss_discr=loss_discr, loss_gen=loss_gen)

            print('   Training finished.')                

    def predict(self, test_gen, tag='pred'):
        pass
        # TODO

    def test(self, test_gen):
        pass
        # TODO
