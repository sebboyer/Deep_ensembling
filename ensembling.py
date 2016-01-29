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

    def __name__(self):
        return "Model"
        
    def train(self,X,y):
        
        # Initiate classifier type
        if self.class_type=="lr": # Sklearn Logistic regression
            estimator=linear_model.LogisticRegression()
            params_to_try = {
                'C': [10**i for i in range(self.Cmin, self.Cmax+1)],
            }
        elif self.class_type=="svm": # Sklearn SVM rbf
            estimator=svm.SVC()
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
            'n_estimators': [100,300,400] ,
            'min_samples_split': [7,10,20],
            'min_samples_leaf': [7,10,20]
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
# - weighted : weighted sum of the results. Need to give weights as inputs 

class Vote:
    def __init__(self,vote_type,weights=[]):
        self.vote_type=vote_type
        self.weights = weights

    def __name__(self):
        return "Vote"
        
    def predict_proba(self,X):
        
        d=np.shape(X)[1]
        
        if self.vote_type=='norm':
            y_norm=(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
            y_vote=np.array([np.sum(y_norm,axis=1)/float(d)]).T
        elif self.vote_type=='rank':
            y_pred_rank=[ss.rankdata(X[:,i])/float(np.shape(X)[0]) for i in range(d)]
            y_pred_rank=np.array(y_pred_rank).T
            y_vote=np.array([np.sum(y_pred_rank,axis=1)/float(d)]).T
        elif self.vote_type == 'weighted':
            if len(self.weights)!=d:
                print "Wrong number of weights"
            else:
                y_vote = np.array([np.sum(self.weights*X,axis=1)/float(d)]).T
        else:
            y_vote=np.array([np.sum(X,axis=1)/float(d)]).T
        
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
        self.links = {} # {est_id:[list of children column to take into account]}
    
    # Models are functions of type : logreg_cv,nn_cv,rf_cv
    def add_model(self,model):
        self.models.append(model)
    
    # Train models one by one and include estimators in self.estimators
    def train_models(self,X=None,y=None):
        estimators=[]
        for m in self.models:
            if isinstance(m,Model): # If m is of class Model: train + append
                estimator,params=m.train(X,y)
                estimators.append(estimator)
            else: # If m is of class Vote : append it
                estimators.append(m)
        self.add_estimators(estimators)
        return estimators
        
    # Estimators must have a predict_proba methods
    def add_estimators(self,estimators):
        self.estimators.extend(estimators)
    
    def activate(self,X):
        #Trigger all estimators on X
        Y=[]
        if self.links == {}: # If no links specified, fully connected
            for m in self.estimators:
                if type(m)==svm.classes.SVC:
                    y_proba=m.decision_function(X)
                    y=(y_proba-np.min(y_proba))/(np.max(y_proba)-np.min(y_proba))
                else:
                    y=m.predict_proba(X)[:,0]
                Y.append(y)
        else: # If links specified, then apply structure
            for est_id in self.links:
                m = self.estimators[est_id]
                children = self.links[est_id]
                X_feat = X[:,children]
                if type(m)==svm.classes.SVC:
                    y_proba=m.decision_function(X_feat)
                    y=(y_proba-np.min(y_proba))/(np.max(y_proba)-np.min(y_proba))
                else:
                    y=m.predict_proba(X_feat)[:,0]
                Y.append(y)
        Y=np.array(Y).T
        return Y

    def evaluate(self,X_list,y_list):
        est_score={}
        for est in range(len(self.estimators)):
            est_score[est]={}
            for ds in range(len(X_list)):
                Y=self.activate(X_list[ds])[:,est]
                Y=np.array([1-x for x in Y])
                est_score[est][ds]=roc_auc_score(y_list[ds],Y)
        return est_score

    def overweight_best_estimator(self,X_list,y_list):
        # Evaluate estimators
        est_score=self.evaluate(X_list,y_list)

        # Find best estimator
        best_est=-1
        best_score=0
        for k in est_score:
            if np.sum(est_score[k].values())>best_score:
                best_est=k

        # Weight repeat best estimator as many times as the number of other estimators
        estimators=[self.estimators[best_est]]*((len(self.estimators)-1)-1)
        self.add_estimators(estimators)


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
        layer=Layer(name)
        for m in models:
            layer.add_model(m)
        self.layers.append(layer)
        
    def train_first_layer(self):
        for i in range(len(self.X_sets)):
            X=self.X_sets[i]
            y=self.y_sets[i]
            estimators=self.layers[0].train_models(X,y)
        return 0

    def train_NonFirst_layer(self,layer_n,X=None,y=None):
        X_in = X
        if X_in != None:
            for i in range(layer_n):
                X_in=self.activate_layer(i,X_in)
        self.layers[layer_n].train_models(X_in,y)

    def train(self,X=None,y=None):
        self.train_first_layer()
        if len(self.layers)>1:
            for i in range(1,len(self.layers)):
                self.train_NonFirst_layer(i,X,y)
        return 0
    
    def activate_layer(self,layer_n,X_in):
        layer=self.layers[layer_n]
        X_out=layer.activate(X_in)
        return X_out   

    def predict_proba(self,X):
        X_in=X
        for i in range(len(self.layers)):
            X_in=self.activate_layer(i,X_in)
        return X_in

    def overweight_best_estimator(self):
        self.layers[0].overweight_best_estimator(self.X_sets,self.y_sets)

    def evaluate(self,X_test,y_test):
        auc={}
        X_in=X_test
        for i in range(len(self.layers)):
            layer_name="Layer "+str(i)
            auc[layer_name]={}
            X_in=self.activate_layer(i,X_in)
            for j in range(np.shape(X_in)[1]):
                Y=np.array([1-x for x in X_in[:,j]])
                estimator_name="Estimator "+str(j)
                auc[layer_name][estimator_name]=roc_auc_score(y_test,Y)
        return auc




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

# Test a single estimator on X_test,y_test
# Returns ROC-AUC
def test_single_model(estimator,X_test,y_test):
    
    if type(estimator)==svm.classes.SVC:
        y_proba=estimator.decision_function(X_test)
        y=(y_proba-np.min(y_proba))/(np.max(y_proba)-np.min(y_proba))
        roc=roc_auc_score(y_test,y)
    else:
        y=estimator.predict_proba(X_test)
        roc=roc_auc_score(y_test,y[:,1])
        
    return roc   

# Train and test classifier of type class_type on the concatenated data from source_list
def test_algo(source_list,target,X_train_list,y_train_list,X_test_list,y_test_list,class_type,Cmin,Cmax):

    # Prepare train data
    X_train,y_train=concatenate(source_list,X_train_list,y_train_list)

    # Prepare test data
    X_test,y_test=X_test_list[target],y_test_list[target]

    # Use GridSearch and cross-validation on train to find best estimator
    m=Model(class_type,Cmin,Cmax)
    estimator,best_params=m.train(X_train,y_train)

    # Compute ROC on test
    roc = test_single_model(estimator,X_test,y_test)
    
    return roc,estimator,best_params











