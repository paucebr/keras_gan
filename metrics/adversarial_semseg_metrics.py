import os
from tools.two_image_iterator import Two_Image_Iterator
from tools.adversarial_batches import *

# mIoU is not tested yet
def get_mIoU(pred, gt_one_hot):
    aux_pred = pred.transpose((0,3,1,2))
    aux_gt = gt_one_hot.transpose((0,3,1,2))

    mIoU = np.zeros((aux_pred.shape[1],1))
    count_mIoU = np.zeros((aux_pred.shape[1],1))      

    for i in range(aux_pred.shape[0]):
        for j in range(aux_pred.shape[1]):
            m = aux_pred[i][j] > 0.5
            n = aux_gt[i][j]

            I = np.logical_and(m,n)
            U = np.logical_or(m,n)

            new_mIoU = ((np.sum(I)*1.0) / (np.sum(U)*1.0))
            
            if np.isnan(new_mIoU) == False:
                mIoU[j] += new_mIoU
                count_mIoU[j] += 1

    mIoU = mIoU / count_mIoU
    return mIoU


def get_validation_metrics(generator, dataset_path, n_classes, batch_size, target_size):
    mIoU = {"train":[], "valid":[], "test":[]}

    train_path = os.path.join(dataset_path, 'train')
    train_it = Two_Image_Iterator(train_path,batch_size=batch_size, 
                    target_size=target_size)
    mIoU["train"] = get_subset_mIoU(train_it, generator, n_classes)


    valid_path = os.path.join(dataset_path, 'valid')
    valid_it = Two_Image_Iterator(valid_path,batch_size=batch_size, 
                    target_size=target_size)
    mIoU["valid"] = get_subset_mIoU(valid_it, generator, n_classes)
    

    test_path = os.path.join(dataset_path, 'test')
    test_it = Two_Image_Iterator(test_path,batch_size=batch_size, 
                    target_size=target_size)
    mIoU["test"] = get_subset_mIoU(test_it, generator, n_classes)

    return mIoU

def get_subset_mIoU(set_it, generator, n_classes):
    set_mIoU = []
    count_mIoU = []
    for i in range(n_classes):
        set_mIoU.append(0.0)
        count_mIoU.append(0.0)

    while set_it.epochs_completed() < 1:
        input_image, gt_one_hot, y_gen = get_batch_for_generator(set_it, n_classes)    
        pred = generator.predict(input_image)
        mIoU = get_mIoU(pred, gt_one_hot)
        for i in range(n_classes):
            if np.isnan(mIoU[i]) == False:
                set_mIoU[i] = set_mIoU[i] + mIoU[i]
                count_mIoU[i] = count_mIoU[i] + 1.0

    #print [x/y for x, y in zip(set_mIoU, count_mIoU)]
    set_mIoU = [x/y for x, y in zip(set_mIoU, count_mIoU)]
    return set_mIoU
