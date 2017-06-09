from adversarial_tools import *

def get_batch_for_discriminator(train_it, n_classes, generator, mode='mix'):
    if mode=='alternate':
        input_img, x_one_hot, y = get_batch_for_discriminator_alternate(train_it, n_classes, generator)
    elif mode=='mix':
        input_img, x_one_hot, y = get_batch_for_discriminator_mix(train_it, n_classes, generator)
    else:
        raise Exception('Unknown type of batch generation for discriminator '+mode)

    return input_img, x_one_hot, y


# TODO: add fliping groundtruth labels 
def get_batch_for_discriminator_mix(train_it, n_classes, generator, 
                                    label_smoothing=True):
    # half of the batch is calculated and half groundtruth
    input_image, groundtruth = train_it.next()
    input_image = normalize(input_image)

    y = np.zeros((train_it.batch_size,2))
    batch_size = train_it.batch_size
    x1 = to_one_hot(groundtruth[:batch_size/2], n_classes)
    y[:batch_size/2,0] = 1.
    
    x2 = generator.predict(input_image[batch_size/2:])
    if label_smoothing:
        y[batch_size/2:,1] = np.random.uniform(low=0.9, high=1., size=len(x2))
    else:
        y[batch_size/2:,1] = 1.

    x_one_hot = np.vstack([x1,x2])
    return input_image, x_one_hot, y    

#TODO: add fliping groundtruth labels 
def get_batch_for_discriminator_alternate(train_it, n_classes, generator,
                                          label_smoothing=True):
    input_image, groundtruth = train_it.next()
    input_image = normalize(input_image)

    # print('batch {}'.format(data_gen.total_batches_seen))
    # one out of every_n batches is a pair of (image, calculated segmentation), 
    # the rest is (image, groundtruth), starting with this case.
    every_n = 2
    if (train_it.total_batches_seen) % every_n == 0:
        # image and calculated semantic segmentation,
        # even batches
        # print 'calculated'
        x_one_hot = generator.predict(input_image)
        y = np.zeros((train_it.batch_size,2))
        y[:,0] = 1.
    else:
        # image and semantic segmentation groundtruth,
        # first and odd batches
        # print 'gt'
        x = groundtruth
        y = np.zeros((train_it.batch_size,2))
        if label_smoothing:
            y[:,1] = np.random.uniform(low=0.9, high=1., size=train_it.batch_size)
            # no cal posar y[:,0] = 1. - y[:,1] ? no per que  en calcular la 
            # xentropy fariem 0*log ?
        else:
            y[:,1] = 1.
            
        x_one_hot = to_one_hot(x, n_classes)

    return input_image, x_one_hot, y


def get_batch_for_generator(it, n_classes):
    input_image, groundtruth = it.next()
    input_image = normalize(input_image)
    
    y = np.zeros((it.batch_size,2), dtype=np.uint8)
    y[:,0] = 1    
    # image and semantic segmentation groundtruth
    return input_image, to_one_hot(groundtruth, n_classes), y
