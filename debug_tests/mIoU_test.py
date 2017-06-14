#from adversarial_semseg_metrics import get_mIoU
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def to_one_hot(x, n_classes):
    enc = OneHotEncoder(n_values=n_classes+1,dtype=np.float,sparse=False)
    x[x > n_classes] = n_classes
    x_one_hot = enc.fit_transform(x.reshape((x.shape[0], x.shape[1]*x.shape[2])))
    x_one_hot = x_one_hot.reshape((x.shape[0], x.shape[1], x.shape[2], n_classes+1))
    x_one_hot = x_one_hot[:,:,:,:-1] #ojocuidao, no contar voids ni en loss ni en accuracy
    return x_one_hot

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

            print (np.sum(I), np.sum(U))

            new_mIoU = ((np.sum(I)*1.0) / (np.sum(U)*1.0))
            
            if np.isnan(new_mIoU) == False:
                mIoU[j] += new_mIoU
                count_mIoU[j] += 1
        print ""
    
    mIoU = mIoU / count_mIoU
    return mIoU

if __name__ == "__main__":
    max_val = 5
    pred = np.random.randint(max_val, size=(2,5,5))
    pred_one_hot = to_one_hot(pred/1., 5)

    gt = np.random.randint(max_val, size=(2,5,5))
    gt_one_hot = to_one_hot(gt/1., 5)

    
    print pred
    print ""
    print gt
    print ""
    print get_mIoU(pred_one_hot, gt_one_hot)

    