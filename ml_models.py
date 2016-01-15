#!/usr/bin/python

'''
Author Sebastien Boyer
Date : Oct 2015

Script containing the ML models for predictive tasks on MOOC data
To run use : 
python ml_models target_course_id

'''



import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import copy
import sys
from multiprocessing import *

from sklearn import svm,linear_model,neighbors,tree, ensemble,neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.cluster import *
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

import pickle
from transfer import *


# Log reg cross val
# Inputs : train and test set + boundary Cmin and Cmax for 10**Cmin<C<10**Cmax
# Output : best estimator (evaluated on train),best parameters
def logreg_cv(X_train,y_train,Cmin,Cmax):
    metric_score='roc_auc'
    estimator=linear_model.LogisticRegression()
    params_to_try = {
        'C': [10**i for i in range(Cmin, Cmax+1)],
    }
    gs = GridSearchCV(estimator,
                      param_grid=params_to_try,
                      cv=5,
                      scoring=metric_score)
    gs.fit(X_train,  y_train)
    return gs.best_estimator_,gs.best_params_

# Nearest neighbors cross val
# Inputs : train and test set + boundary Cmin and Cmax for 10**Cmin<C<10**Cmax
# Output : best estimator (evaluated on train),best parameters
def nn_cv(X_train,y_train,n_min,n_max):
    metric_score='roc_auc'
    estimator=neighbors.KNeighborsClassifier()
    params_to_try = {
        'n_neighbors': [5*i for i in range(n_min/5, n_max/5+1)],
    }
    gs = GridSearchCV(estimator,
                      param_grid=params_to_try,
                      cv=5,
                      scoring=metric_score)
    gs.fit(X_train,  y_train)
    return gs.best_estimator_,gs.best_params_



# Random forest cross val
# Inputs : train and test set + boundary Cmin and Cmax for 10**Cmin<C<10**Cmax
# Output : best estimator (evaluated on train),best parameters
def rf_cv(X_train,y_train,n_min,n_max):
    metric_score='roc_auc'
    estimator=ensemble.RandomForestClassifier()
    params_to_try = {
    'n_estimators': [50,100,300] ,
    'min_samples_split': [3,7,10],
    'min_samples_leaf': [3,7,10]
    }

    gs = GridSearchCV(estimator,
                      param_grid=params_to_try,
                      cv=5,
                      scoring=metric_score,
                      n_jobs=-1)
    gs.fit(X_train,  y_train)
    return gs.best_estimator_,gs.best_params_




# Test GridSearch approach to Logistic regression
# Output : roc on test, trained estimator, parameters used in best estimator
def test_algo(algo_func,source,target,X_train_list,y_train_list,X_test_list,y_test_list,Cmin,Cmax):

    # If more than one source, concatenate the source, otherwise use source
    if len(source)>1:
        X_train=np.concatenate([X_train_list[c] for c in source],axis=0)
        y_train=np.concatenate([y_train_list[c] for c in source],axis=0)
    else:
        source=source[0]
        X_train,y_train=X_train_list[source],y_train_list[source]

    # Load test data
    X_test,y_test=X_test_list[target],y_test_list[target]

    # Use GridSearch and cross-validation on train to find best estimator
    estimator,best_params=algo_func(X_train,y_train,Cmin,Cmax)

    # Compute ROC on test
    y_proba=estimator.predict_proba(X_test)
    roc=roc_auc_score(y_test,y_proba[:,1])

    return roc,estimator,best_params



# Stacking pre-trained models
# Take pre-trained models, train and test sets
# Returns roc on test of combined pre-trained models using logreg
def stacking_model(model_list,X_train,y_train,X_test,y_test):

    # Train all models on all data
    y_train_list=[]
    y_test_list=[]
    for m in model_list:
        y_train_int=m.predict_proba(X_train)[:,0]
        y_test_int=m.predict_proba(X_test)[:,0]
        y_train_list.append(y_train_int)
        y_test_list.append(y_test_int)
        
    y_train_int=np.array(y_train_list).T
    y_test_int=np.array(y_test_list).T
    
    # Stacking model
    Cmin,Cmax=-6,5
    estimator_stacking,best_params=logreg_cv(y_train_int,y_train,Cmin,Cmax)
    y_proba=estimator_stacking.predict_proba(y_test_int)
    roc=roc_auc_score(y_test,y_proba[:,1])
    
    return roc

# Take pre-trained classification models and train a logreg on top of the ouputs using source as the list of course to train on
# Output the ROC of the high level model tested on the target course
def test_deep(models_pretrained,source,target,X_train_list,y_train_list,X_test_list,y_test_list):
        
    # If more than one source, concatenate the source, otherwise use source
    if len(source)>1:
        X_train=np.concatenate([X_train_list[c] for c in source],axis=0)
        y_train=np.concatenate([y_train_list[c] for c in source],axis=0)
    else:
        source=source[0]
        X_train,y_train=X_train_list[source],y_train_list[source]
        
    roc=stacking_model(models_pretrained,X_train,y_train,X_test_list[target],y_test_list[target])

    return roc

# Take data and source_list and target, Trained via cross valid and test roc with different methods
# Return dictionnary of results

def generateAll_roc(source_list,target,X_train_list,y_train_list,X_test_list,y_test_list):
    d={}
    models=[]
    Cmin,Cmax=1,6
    for s in source_list:
        ############ Source  = s
        source=[s]
        # Logreg
        Cmin,Cmax=1,6
        roc,estimator_lr,best_params=test_algo(logreg_cv,source,target,X_train_list,y_train_list,X_test_list,y_test_list,Cmin,Cmax)
        models.append(estimator_lr)
        d['Logreg_s'+str(s)]=roc
        # NN
        Cmin,Cmax=110,140
        roc,estimator_nn,best_params=test_algo(nn_cv,source,target,X_train_list,y_train_list,X_test_list,y_test_list,Cmin,Cmax)
        models.append(estimator_nn)
        d['NN_s'+str(s)]=roc

    ############## Source = Target
    # Logreg
    source=[target]
    Cmin,Cmax=1,6
    roc,estimator_lr,best_params=test_algo(logreg_cv,source,target,X_train_list,y_train_list,X_test_list,y_test_list,Cmin,Cmax)
    d['Logreg_self']=roc
        
    ############## Source  = Concat
    Cmin,Cmax=0,6
    roc,estimator_lr_concat,best_params=test_algo(logreg_cv,source_list,target,X_train_list,y_train_list,X_test_list,y_test_list,Cmin,Cmax)
    models.append(estimator_lr_concat)
    d['Logreg_concat']=roc

    Cmin,Cmax=110,140
    roc,estimator_nn_concat,best_params=test_algo(nn_cv,source_list,target,X_train_list,y_train_list,X_test_list,y_test_list,Cmin,Cmax)
    models.append(estimator_nn_concat)
    d['NN_concat']=roc

    ############## Stacking model "+str(len(models))+" models used.
    roc=test_deep(models,source_list,target,X_train_list,y_train_list,X_test_list,y_test_list)
    d['Stacking']=roc
    
    return d

# Take dictionnary and general ROC for all methods (using generateAll_roc) and for all problems over the first 14 weeks
# Returns the dictionnary of results
def generateAll_roc_AllProblems(source_list,target,feat_dic_list,featids_feat):
    
    results={}
    pool = Pool()
    setProblems=[]
    
    # Initialize problems
    for target_week in np.arange(2,14):
        for feat_week in np.arange(1,target_week):
            setProblems.append((target_week,feat_week))
    
    # Run problem experiments in parallel
    results_list = pool.map(run_problem, setProblems)

    # Change results format from list to dictionnary
    for i,pbid in enumerate(setProblems):
        results[pbid]=results_list[i]
    
    # Dumping results in .p file
    name="results_"+str(source_list)+"_"+str(target)+".p"
    pickle.dump( results, open( name, "wb" ) )
    
    return results

# Parallel process - on problem pbid=(target_week,feat_week)
# Returns dictionnary of results for this problem
def run_problem(pbid):
        target_week=pbid[0]
        feat_week=pbid[1]

        print "################ Problem : target_week=",target_week," feature weeks = ",feat_week

        # Defining problem
        feat_weekids=[feat_week]
        label_weekid=target_week

        # Preparing data
        train_ratio=0.8
        X_train_list,X_test_list,y_train_list,y_test_list=prepare_problem_data_all(feat_dic_list,featids_feat,feat_weekids,label_weekid,train_ratio)

        # Computing results
        r=generateAll_roc(source_list,target,X_train_list,y_train_list,X_test_list,y_test_list)

        return r


if __name__ == "__main__":

    ######## Take arguments :
    target=int(sys.argv[1])
    source_list=list(set([0,1,3,4]) - set([target]))

    ######## Data Feature acquisition from file 
    print " ########################## Downloading data from .csv files"
    filename1='6002x13_user_longitudinal_feature_values.csv'
    filename2='6002x12_user_longitudinal_feature_values.csv'
    filename3='1473x_user_longitudinal_feature_values.csv'
    filename4='201x13_user_longitudinal_feature_values.csv'
    filename5='3091x12_user_longitudinal_feature_values.csv'
    filename6='3091x13_user_longitudinal_feature_values.csv'

    files=[filename1,filename2,filename3,filename4,filename5,filename6]
    n=len(files)
    feat_dic_list,featids_feat,count_duplicate_list=extract_structured_data_from_file(files)

    ######### Generate All ROC for this source_list and target
    print " ############################ Running experiments "
    results=generateAll_roc_AllProblems(source_list,target,feat_dic_list,featids_feat)

    ######### Dump this result in a pickle file
    print " ############################ Dumping results "
    name="results_"+str(source_list)+"_"+str(target)+".p"
    pickle.dump( results, open( name, "wb" ) )
    print " ###################### This is a success !! ##################### "











