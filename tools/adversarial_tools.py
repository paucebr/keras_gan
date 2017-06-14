# Imports
import os
import numpy as np
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


dim_ordering = K.image_dim_ordering()

""" simple normalization of the images """        
def normalize(input_image):
    return input_image/255.


"""  transform class labels to one-hot encoding but taking care of the
void class, which we remove """
def to_one_hot(x, n_classes):
    enc = OneHotEncoder(n_values=n_classes+1,dtype=np.float,sparse=False)
    x[x > n_classes] = n_classes
    x_one_hot = enc.fit_transform(x.reshape((x.shape[0], x.shape[1]*x.shape[2])))
    x_one_hot = x_one_hot.reshape((x.shape[0], x.shape[1], x.shape[2], n_classes+1))
    x_one_hot = x_one_hot[:,:,:,:-1] #ojocuidao, no contar voids ni en loss ni en accuracy
    return x_one_hot

""" This is fix the generator when training the discriminator, and the 
opposite. I don't know if it is equivalent to do net.trainable = True 
but I do this way just in case. """
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

#be carefull, we only work with tensorflow
def channel_idx():
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        return 1
    else:
        return 3


def plot_loss2(loss_gen, loss_discr, savepath, loss_weights):
    #warning, integrate comments, delete everithing else
    
    #1 - Losses Plot
    plt.figure(figsize=(10, 8))
    plt.plot(loss_discr, label='discriminitive loss')
    plt.plot(loss_gen, label='generative loss')
    plt.legend()
    
    # Save fig
    plt.savefig(os.path.join(savepath, 'plot_loss.png'))
    plt.close()


    #plt.figure('loss_discriminator')
    #plt.plot(loss_discr,label='discriminator')
    #plt.plot(loss_gen[:,2], label='gen. cheat discr.')
    #plt.legend()

    #plt.figure('loss_generator')
    #plt.plot(loss_gen[:,0], label='total generator')
    #plt.plot(loss_gen[:,1]*gan_loss_weights[0], label='gen. semantic seg.')
    #plt.plot(loss_gen[:,2]*gan_loss_weights[1], label='gen. cheat discr.')
    #plt.legend()

def plot_loss(loss_gen, loss_discr, savepath, loss_weights):
    plt.figure('loss_discriminator')
    plt.plot(loss_discr,label='discriminator')
    plt.plot(loss_gen[:,2], label='gen. cheat discr.')
    plt.legend()

    plt.figure('loss_generator')
    plt.plot(loss_gen[:,0], label='total generator')
    plt.plot(loss_gen[:,1]*loss_weights[0], label='gen. semantic seg.')
    plt.plot(loss_gen[:,2]*loss_weights[1], label='gen. cheat discr.')
    plt.legend()

    plt.savefig(os.path.join(savepath, 'plot_loss.png'))
    plt.close()

def plot_accuracy(accuracy, classes, n_classes, savepath, color_map, classes_per_plot=4):
    #2 - mIoU Plot
    aux_color = []
    for x in range(len(color_map)):
        color = [y/255. for y in color_map[x]]
        aux_color.append(color)

    for i in range(n_classes/classes_per_plot):
        plt.figure(figsize=(10, 8))
        for j in range(classes_per_plot):
            plt.plot(accuracy["train"][i*classes_per_plot+j], label=classes[i*classes_per_plot+j]+"_train", color=aux_color[i*classes_per_plot+j])
            plt.plot(accuracy["valid"][i*classes_per_plot+j], label=classes[i*classes_per_plot+j]+"_valid", color=aux_color[i*classes_per_plot+j], ls="--")
            plt.plot(accuracy["test"][i*classes_per_plot+j], label=classes[i*classes_per_plot+j]+"_test",color=aux_color[i*classes_per_plot+j], ls="dotted")
        
        plt.legend()
        plt.savefig(os.path.join(savepath, 'plot_mIoU_'+str(i)+'.png'))
        plt.close()

    #this is awfull, how do we show the mod clases?
    if ((n_classes/classes_per_plot) < (n_classes/(classes_per_plot*1.0))):
        i=i+1
        plt.figure(figsize=(10, 8))
        for j in range(n_classes%classes_per_plot):
            plt.plot(accuracy["train"][i*classes_per_plot+j], label=classes[i*classes_per_plot+j]+"_train", color=aux_color[i*classes_per_plot+j])
            plt.plot(accuracy["valid"][i*classes_per_plot+j], label=classes[i*classes_per_plot+j]+"_valid", color=aux_color[i*classes_per_plot+j], ls="--")
            plt.plot(accuracy["test"][i*classes_per_plot+j], label=classes[i*classes_per_plot+j]+"_test",color=aux_color[i*classes_per_plot+j], ls="dotted")
        
        plt.legend()
        plt.savefig(os.path.join(savepath, 'plot_mIoU_'+str(i)+'.png'))
        plt.close()