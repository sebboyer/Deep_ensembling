import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import copy
import pandas as pd
from sklearn.metrics import *



def plot_feature_matrix(n_top,X_train1,X_train2,y_train1,y_train2):

    y_train1_pos=np.array([y_train1]).T[np.argsort(-y_train1)[:n_top]]
    y_train1_neg=np.array([y_train1]).T[np.argsort(y_train1)[:n_top]]
    X_train1_neg=X_train1[np.argsort(y_train1)[:n_top],:]
    X_train1_pos=X_train1[np.argsort(-y_train1)[:n_top],:]

    y_train2_pos=np.array([y_train2]).T[np.argsort(-y_train2)[:n_top]]
    y_train2_neg=np.array([y_train2]).T[np.argsort(y_train2)[:n_top]]
    X_train2_neg=X_train2[np.argsort(y_train2)[:n_top],:]
    X_train2_pos=X_train2[np.argsort(-y_train2)[:n_top],:]

    mat1_pos=np.concatenate([X_train1_pos,y_train1_pos],axis=1)
    mat2_pos=np.concatenate([X_train2_pos,y_train2_pos],axis=1)
    mat1_neg=np.concatenate([X_train1_neg,y_train1_neg],axis=1)
    mat2_neg=np.concatenate([X_train2_neg,y_train2_neg],axis=1)

    mat1=np.concatenate([mat1_pos,mat1_neg],axis=0)
    mat2=np.concatenate([mat2_pos,mat2_neg],axis=0)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,20))
    cax = ax1.imshow(mat1, interpolation='nearest', cmap=plt.cm.coolwarm)
    cax = ax2.imshow(mat2, interpolation='nearest', cmap=plt.cm.coolwarm)
    ax1.set_title('Features and labels 1')
    ax2.set_title('Features and labels 2')
    plt.show()

def plot_size_summaries(y_train_list,files,feat_weekids,label_weekid):

    n=len(y_train_list)
    neg=[y_train_list[i][y_train_list[i]==0].shape[0] for i in range(n)]
    pos=[y_train_list[i][y_train_list[i]==1].shape[0] for i in range(n)]

    fig, ax = plt.subplots(figsize=(10,5))

    ind=np.arange(n)
    width=0.4
    labels=[y[:7] for y in files]

    ax.bar(ind,neg,width)
    ax.bar(ind+0.2,pos,width,color='r')

    ax.set_xticks(ind + width/1.3)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_title('Global statistics about Problem : '+str(feat_weekids)+" predict "+str(label_weekid),fontsize=18)
    ax.set_ylabel('Dropout (blue) ; Non-Dropout (red)',fontsize=13)
    plt.show()



# Return pandas DF for TrueP and FalseP used in the ROC AUC for the estimator on the X_test,y_test data
def TPR_FRP(estimator, X_test, y_test,cat):

    # Convert y_test to right format for roc_curve
    y_test_array=np.zeros((np.shape(y_test)[0],1))
    y_test_array[:,0]=y_test
    
    # Test estimator on test data
    Y_proba=estimator.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_test_array[:, i], Y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_proba[:,0].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    category=[cat]*len(tpr[0])
    # Formatting using pandas DF
    d={'TRP':tpr[0],'FRP':fpr[0],'model':category}
    data=pd.DataFrame.from_dict(d)

    return data















































