'''
Author : Sebastien Boyer
Date : January 2016


Train multiple models on multiple DataSets

Models currently supported :
- 'lr' for Logistic Regresssion
- 'nn' for KNearestNeighbors
- 'rf' for Random forest
- 'svm' for SupportVectorMachine


Model type can be one of the two :
- 'ind' train individual models on each DataSet
- 'concat' train models on the concatenation of all DataSet but one

'''


#from dataFormat import *
from ensembling import *


import numpy as np
import pickle
from multiprocessing import *
from functools import partial
import sys


def train_individual_models(X_train_list,y_train_list,model_list,params_list,para = 0):

    # Initialize results
    dic={}

    # Train models in parallel for each source courses
    list_sources = range(len(X_train_list))
    if para==1:
        pool = Pool()
        partial_train_model = partial(train_model,X_train_list,X_train_list,y_train_list,y_train_list,model_list,params_list)
        estimators_list = pool.map(partial_train_model, list_sources)
    else:
        estimators_list = []
        for source in list_sources:
            estimators_list.append(train_model(X_train_list,X_train_list,y_train_list,y_train_list,model_list,params_list,source))

    # Transform list into dictionnary
    for i,estimator_dic in enumerate(estimators_list):
        dic[i] = estimator_dic

    return dic


def train_model(X_train_list,X_test_list,y_train_list,y_test_list,model_list,params_list,source):

    print "##### Train for DataSet : ",source
    dic={}
    source=[source]
    target=0 # Target doesn't matter because we don't remember evaluation (only estimator)

    for i,model in enumerate(model_list):
        print "# Training "+model
        Cmin,Cmax=params_list[i]
        roc,est,best_params=test_algo(source,target,X_train_list,y_train_list,X_test_list,y_test_list,model,Cmin,Cmax)
        dic[model]=est

    return dic


def train_concatenated_models(X_train_list,y_train_list,model_list,params_list,para = 0):

    # Initialize results
    dic={}

    # Train models in parallel for each source courses
    all_dataSets = range(len(X_train_list))
    all_dataSets_mult = [all_dataSets for i in all_dataSets]
    list_sources = [[i for i in all_dataSets_mult[j] if i!=j] for j in all_dataSets]

    if para==1:
        pool = Pool()
        partial_train_model = partial(train_concat_model,X_train_list,X_train_list,y_train_list,y_train_list,model_list,params_list)
        estimators_list = pool.map(partial_train_model, list_sources)
    else:
        estimators_list = []
        for sources in list_sources:
            estimators_list.append(train_concat_model(X_train_list,X_train_list,y_train_list,y_train_list,model_list,params_list,sources))

    # Transform list into dictionnary
    for i,estimator_dic in enumerate(estimators_list):
        dic[i] = estimator_dic

    return dic

def train_concat_model(X_train_list,X_test_list,y_train_list,y_test_list,model_list,params_list,sources):

    print "##### Train for DataSets : ",sources
    dic={}
    target = sources[0] # We don't remember AUC anyways

    for i,model in enumerate(model_list):
        print "# Training "+model
        Cmin,Cmax=params_list[i]
        roc,est,best_params=test_algo(sources,target,X_train_list,y_train_list,X_test_list,y_test_list,model,Cmin,Cmax)
        dic[model]=est

    return dic


def main(X_train_list,y_train_list,model_list,params_list,model_type='ind',output_filename='models.p',pic = 0):


    train_func = 0
    if model_type=='concat':
        train_func = train_concatenated_models
    else:
        train_func = train_individual_models


    dic=train_func(X_train_list,y_train_list,model_list,params_list)

    if pic==1:
        pickle.dump(dic,open( output_filename, "wb" ) )

    return dic
