import os
import numpy as np
import random
# Keras imports
import keras.models as kmodels
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.utils.visualize_util import plot
from model import Model
from generator import build_generator
from segnet import build_segnet
from unet import build_unet
from tqdm import tqdm
import matplotlib.pyplot as plt
from two_image_iterator import Two_Image_Iterator
from sklearn.preprocessing import OneHotEncoder
from tools.save_images import save_img3
import math



class Basic_Semseg(Model):
    def __init__(self, cf, img_shape):
        self.cf = cf
        self.img_shape = img_shape
        self.n_classes = self.cf.dataset.n_classes
        self.optimizer = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08) 
        self.generator = self.make_generator(self.img_shape,
                                             self.cf.dataset.n_classes,
                                             self.optimizer,
                                             the_loss='categorical_crossentropy',
                                             metrics=[])



    # Make generator
    def make_generator(self, img_shape, n_classes, optimizer,
                       the_loss='categorical_crossentropy', metrics=[]):
        # Build model
        # generator = build_generator(img_shape, n_channels=200, l2_reg=0.)
        generator = build_segnet(img_shape, n_classes=n_classes, l2_reg=0.)

        #EXCHALE UN OJO A IMG_SHAPE, A VER QUE ESTAS METIENDO POR AQUI

        # Compile model
        generator.compile(loss=the_loss, metrics=metrics, optimizer=optimizer)

        # Show model
        if self.cf.show_model:
            print('Generator')
            generator.summary()
            plot(generator,
                 to_file=os.path.join(self.cf.savepath, 'model_generator.png'))
        return generator

    def train(self, train_gen, valid_gen, cb):
        if (self.cf.train_model):
            losses = {"g": [], "mIoU":[]}

            for i in range(self.cf.dataset.n_classes):
                losses["mIoU"].append([])

            n_epochs = self.cf.n_epochs
            batch_size = self.cf.batch_size_train
            #img_shape = self.model.img_shape
            #n_classes = self.model.n_classes
            img_shape = (256, 256, 3)
            n_classes = self.cf.dataset.n_classes

            save_path = self.cf.savepath
            tag = 'train'
            color_map = self.cf.dataset.color_map
            saveResults = True
            display_every_batches = 200
            save_every_batches = 200
            void_label = self.cf.dataset.void_class

            train_it = Two_Image_Iterator('/datatmp/Datasets/segmentation/cityscapes/train/',batch_size=batch_size, 
                        target_size=img_shape[:-1])

            valid_it = Two_Image_Iterator('/datatmp/Datasets/segmentation/cityscapes/valid/',batch_size=batch_size, 
                        target_size=img_shape[:-1])

            generator_model = self.generator


            while train_it.epochs_completed() < n_epochs:

                input_image, gt_one_hot, y_gen = get_batch_for_generator(train_it, n_classes)                   
                lg = generator_model.train_on_batch(input_image, gt_one_hot)

                if train_it.total_batches_seen % display_every_batches == 0:
                    input_image_valid, gt_one_hot_valid, y_gen_valid = get_batch_for_generator(valid_it, n_classes)    
                    pred_valid = generator_model.predict(input_image_valid)
                    mIoU = self.get_mIoU(pred_valid, gt_one_hot_valid)
                    losses["g"].append(lg)
                    for i in range(self.cf.dataset.n_classes):
                        losses["mIoU"][i].append(mIoU[i])

                    #np.append(losses["mIoU"], mIoU, axis=1)
                    print('epoch {}, batch {}, loss generator {}, mIoU {}'.\
                    format(train_it.epochs_completed(), train_it.total_batches_seen, lg, mIoU.tolist()))
                    self.plot_loss(losses)

                if saveResults and train_it.total_batches_seen % save_every_batches == 0:
                    pred = generator_model.predict(input_image)
                    save_img3(input_image*255, gt_one_hot, pred, save_path, train_it.epochs_completed(),
                      color_map, n_classes, tag+str(train_it.epochs_completed()),
                      void_label)