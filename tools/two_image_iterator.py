# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:05:56 2017

@author: joans

A Two_Image_Iterator object browses a pair of folders where corresponding
images have the same name. The method next() returns on such pair. The idea
is to get an image and its semantic segmentation labeling.
"""

import os
import random
import numpy as np

from keras.preprocessing.image import Iterator, load_img, img_to_array

class Two_Image_Iterator(Iterator):
    """Class to iterate A and B images at the same time."""

    def __init__(self, directory, a_dir_name='images', b_dir_name='masks', 
                 is_a_grayscale=False, is_b_grayscale=True, 
                 target_size=(256, 256), dim_ordering='tf', N=-1,
                 batch_size=32, shuffle=True, seed=None):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - is_a_grayscale: if True, A images will only have one channel.
        - is_b_grayscale: if True, B images will only have one channel.
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator.
        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.b_dir = os.path.join(directory, b_dir_name)

        a_files = set(x for x in os.listdir(self.a_dir))
        b_files = set(x for x in os.listdir(self.b_dir))
        # Files inside a and b should have the same name. Images without a pair are discarded.
        self.filenames = list(a_files.intersection(b_files))

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)
        if self.N == 0:
            raise Exception("""Did not find any pair in the dataset. Please check that """
                            """the names and extensions of the pairs are exactly the same. """
                            """Searched inside folders: {0} and {1}""".format(self.a_dir, self.b_dir))
                            
        self.dim_ordering = dim_ordering
        if self.dim_ordering not in ('th', 'default', 'tf'):
            raise Exception('dim_ordering should be one of "th", "tf" or "default". '
                            'Got {0}'.format(self.dim_ordering))

        self.target_size = target_size

        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale

        self.image_shape_a = self._get_image_shape(self.is_a_grayscale)
        self.image_shape_b = self._get_image_shape(self.is_b_grayscale)

        if self.dim_ordering in ('th', 'default'):
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        super(Two_Image_Iterator, self).__init__(len(self.filenames), batch_size,
                                               shuffle, seed)

    def _get_image_shape(self, is_grayscale):
        """Auxiliar method to get the image shape given the color mode."""
        if is_grayscale:
            if self.dim_ordering == 'tf':
                return self.target_size + (1,)
            else:
                return (1,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                return self.target_size + (3,)
            else:
                return (3,) + self.target_size


    def _load_img_pair(self, idx, load_from_memory):
        """Get a pair of images with index idx."""
        if load_from_memory:
            a = self.a[idx]
            b = self.b[idx]
            return a, b

        fname = self.filenames[idx]

        a = load_img(os.path.join(self.a_dir, fname),
                     grayscale=self.is_a_grayscale,
                     target_size=self.target_size)
        b = load_img(os.path.join(self.b_dir, fname),
                     grayscale=self.is_b_grayscale,
                     target_size=self.target_size)

        a = img_to_array(a, self.dim_ordering)
        b = img_to_array(b, self.dim_ordering)

        return a, b


    def epochs_completed(self):
        return (self.total_batches_seen*self.batch_size)/self.N

    def next(self):
        self.load_from_memory = False
        
        """Get the next pair of the sequence."""
        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
                    
        # this is to avoid returning a batch with less samples than batch_size
        if current_batch_size<self.batch_size:
            self.total_batches_seen -= 1
            random.shuffle(self.filenames)
            super(Two_Image_Iterator, self).reset()
            return self.next()

        batch_a = np.zeros((current_batch_size,) + self.image_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.image_shape_b)

        for i, j in enumerate(index_array):
            a_img, b_img = self._load_img_pair(j, self.load_from_memory)
            
            batch_a[i] = a_img
            batch_b[i] = b_img

        return [batch_a, batch_b]
        
        
        
    def reset(self):
        self.total_batches_seen = 0
        random.shuffle(self.filenames)
        super(Two_Image_Iterator, self).reset()
        
        


if __name__ == "__main__":   
    import matplotlib.pyplot as plt

    train_it = Two_Image_Iterator('/data/Datasets/segmentation/camvid/train/', 
                                  batch_size=32, target_size=(256,256))
    while train_it.epochs_completed() < 2:
        a,b = train_it.next()
        print(('batches seen {}, batch_index {}, epochs completed {}, '\
              + 'number of samples {}').format(train_it.total_batches_seen, \
              train_it.batch_index, train_it.epochs_completed(), len(a)))
    
    plt.figure()        
    plt.imshow(a[0].astype(np.uint8))
    plt.figure()
    plt.imshow(np.squeeze(b[0],axis=2).astype(np.uint8))
        
    train_it.reset() # ready to start again drawing batches for a new training session
    