'''

Ensembling framework

Author : Sebastien Boyer
Date : January 2016

'''

import numpy as np

from sklearn import svm,linear_model,neighbors,tree, ensemble,neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.cluster import *
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

import scipy.stats as ss



###############################################################################
################################# Models ######################################
###############################################################################


class Model:
    def __init__(self,class_type,Cmin,Cmax):
        self.class_type=class_type # One of : lr, nn, rf
        self.Cmin=Cmin
        self.Cmax=Cmax
        
    def train(self,X,y):
        
        # Initiate classifier type
        if self.class_type=="lr": # Sklearn Logistic regression
            estimator=linear_model.LogisticRegression()
            params_to_try = {
                'C': [10**i for i in range(self.Cmin, self.Cmax+1)],
            }
        elif self.class_type=="nn": # Sklearn KNN
            estimator=neighbors.KNeighborsClassifier()
            params_to_try = {
            'n_neighbors': [5*i for i in range(self.Cmin/5, self.Cmax/5+1)],
            }
        elif self.class_type=="rf": # Sklearn Random forest
            estimator=ensemble.RandomForestClassifier()
            params_to_try = {
            'n_estimators': [50,100,300] ,
            'min_samples_split': [3,7,10],
            'min_samples_leaf': [3,7,10]
            }
        else: # Add new classifier here
            print "Wrong classifier type. Must be one of : lr, nn, rf"
         
        # Find optimal parameters through 5-fold cross validation
        metric_score='roc_auc'
        gs = GridSearchCV(estimator,
                          param_grid=params_to_try,
                          cv=5,
                          scoring=metric_score)
        gs.fit(X,  y)
        
        # Return best estimator and best params
        return gs.best_estimator_,gs.best_params_




###############################################################################
################################# Votes #######################################
###############################################################################

# This class contains the vote paradigm to "blend" results from classifiers
# 3 types of votes :
# - simple : sum the results
# - norm : normalize the results of each classifier so that min=0 and max=1 and then sum
# - rank : rank each probability inside each classifier and average them

class Vote:
    def __init__(self,vote_type):
        self.vote_type=vote_type
        
    def predict_proba(self,X):
        
        d=np.shape(X)[1]
        
        if self.vote_type=='norm':
            y_norm=(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
            y_vote=np.array([1-np.sum(y_norm,axis=1)/float(d)]).T
        elif self.vote_type=='rank':
            y_pred_rank=[ss.rankdata(X[:,i])/float(np.shape(X)[0]) for i in range(d)]
            y_pred_rank=np.array(y_pred_rank).T
            y_vote=np.array([1-np.sum(y_pred_rank,axis=1)/float(d)]).T
        else:
            y_vote=np.array([1-np.sum(X,axis=1)/float(d)]).T
        
        return y_vote
        


###############################################################################
################################# Layers ######################################
###############################################################################

# This class contains the notion of Layer. A layer contains a set of models (that can be trained into estimators) and a set of estimators (trained models OR vote paradigms)

class Layer:
    def __init__(self,name):
        self.name=name
        self.models=[]
        self.estimators=[]  # Trained models or votes
    
    # Models are functions of type : logreg_cv,nn_cv,rf_cv
    def add_model(self,model):
        self.models.append(model)
    
    # Train models one by one and include estimators in self.estimators
    def train_models(self,X,y):
        estimators=[]
        for m in self.models:
            estimator,params=m.train(X,y)
            estimators.append(estimator)
        return estimators
        
    # Estimators must have a predict_proba methods
    def add_estimators(self,estimators):
        self.estimators.extend(estimators)
    
    def activate(self,X):
        #Trigger all estimators on X
        Y=[]
        for m in self.estimators:
            y=m.predict_proba(X)[:,0]
            Y.append(y)
        Y=np.array(Y).T
        return Y


###############################################################################
################################# Network #####################################
###############################################################################

# This class contains the Network class. A Network is composed of a set of training sets and a set of layers that list the models used and that states the structure with which models are merged together to produce the final output

class Network:
    def __init__(self):
        self.layers=[]
        self.X_sets=[]
        self.y_sets=[]
        
    def add_trainSet(self,X,y):
        if np.shape(X)[0]!=np.shape(y)[0]:
            print "Dimensions of X and y don't match"
            return 1
        self.X_sets.append(X)
        self.y_sets.append(y)
        return 0
        
    def add_layer(self,name,models): 
        layer=Layer2(name)
        for m in models:
            layer.add_model(m)
        self.layers.append(layer)
        
    def train_first_layer(self):
        for i in range(len(self.X_sets)):
            X=self.X_sets[i]
            y=self.y_sets[i]
            estimators=self.layers[0].train_models(X,y)
            self.layers[0].add_estimators(estimators)
        return 0
    
    def activate_layer(self,layer_n,X_in):
        # Activate first layer
        layer=self.layers[layer_n]
        X_out=layer.activate(X_in)
        return X_out                



###############################################################################
################################# Functions ###################################
###############################################################################

# Concatenate the dataset at the index positions source_list in X_train_list and y_train_list
def concatenate(source_list,X_train_list,y_train_list):
    if len(source_list)>1:
        X_train=np.concatenate([X_train_list[c] for c in source_list],axis=0)
        y_train=np.concatenate([y_train_list[c] for c in source_list],axis=0)
    else:
        source=source_list[0]
        X_train,y_train=X_train_list[source],y_train_list[source]
        
    return X_train,y_train

# Train and test classifier of type class_type on the concatenated data from source_list
def test_algo(source_list,target,X_train_list,y_train_list,X_test_list,y_test_list,class_type,Cmin,Cmax):

    # Prepare train data
    X_train,y_train=concatenate(source_list,X_train_list,y_train_list)

    # Prepare test data
    X_test,y_test=X_test_list[target],y_test_list[target]

    # Use GridSearch and cross-validation on train to find best estimator
    m=Model(class_type,Cmin,Cmax)
    m.train(X_train,y_train)
    estimator,best_params=m.estimator,m.best_params

    # Compute ROC on test
    y_proba=estimator.predict_proba(X_test)
    roc=roc_auc_score(y_test,y_proba[:,1])

    return roc,m









