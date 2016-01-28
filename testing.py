'''
Author : Sebastien Boyer
Date : January 2016

Test models 

'''



############# Dependencies
import sys
from dataFormat import *
from visua import *
from ensembling import *

import numpy as np
from ggplot import *

from multiprocessing import *
from functools import partial



###############################################################################
########################## Network Creation ###################################
###############################################################################
'''
Return a simple 3 Layers network build from the given estimators
'''

def create_simple_network(estimators):
    
    ### Define Network
    N=Network()
    # First layer
    N.add_layer("Models_layer",[])
    N.layers[0].add_estimators(estimators)
    # Second Layer
    N.add_layer("Hidden_layer",[Vote("simple"),Vote("rank"),Vote("norm")])
    # Third Layer
    N.add_layer("Output_layer",[Vote("simple")])

    # Train Network (nothing here because only Vote models)
    N.train()
    
    return N

'''
Return a 3 Layers network whose best estimator (only trainset) is overweitghed
'''
def create_weighted_network(X_train,y_train,estimators):
    
    ### Define Network
    N=Network()
    # First layer
    N.add_layer("Models_layer",[])
    N.layers[0].add_estimators(estimators)
    # Second Layer
    N.add_layer("Hidden_layer",[Vote("simple"),Vote("rank"),Vote("norm")])
    # Third Layer
    N.add_layer("Output_layer",[Vote("simple")])

    ### Add train data to Network
    N.add_trainSet(X_train,y_train)

    ### Repeat best estimator on training data
    N.overweight_best_estimator()

    # Train Network (nothing here because only Vote models)
    N.train()
    
    return N





###############################################################################
####################### Test models and Networks ##############################
###############################################################################
'''
Test models in individual_models for all DataSet in X_test_list, y_test_list
Args :  models, list of problems (target week, feature week), structure of network to use, list of X, list of y , list of models to skip in ['lr','svm','nn','rf']
Returns : results[pb][target][source] contains the corresponding AUC on test for the given network of estimators
'''
def test_all_models(individual_models,X_test_list,y_test_list,DataSets=[],create_net_func=create_simple_network,para=0,skip_models=[]):

    if DataSets==[]:
        DataSets=range(len(X_test_list))

    ##### Initialize results
    results={}
    for target in DataSets: # For all possible target DataSets in DataSets
        results[target]={}
        potential_sources = [c for c in DataSets if c!=target]+['all']
        for source in potential_sources: # For all remaning source as source including the aggregation of all
            results[target][source] = {}

    ##### Fill results
    if para==1:
        pool = Pool()
        partial_test_model = partial(test_model,individual_models,DataSets,create_net_func,X_test_list,y_test_list,skip_models)
        estimators_list = pool.map(partial_test_model, DataSets)
    else:
        estimators_list = []
        for target in DataSets:
            estimators_list.append(test_model(individual_models,DataSets,create_net_func,X_test_list,y_test_list,skip_models,target))
    
    # Transform list into dictionnary
    for i,estimator_dic in enumerate(estimators_list):
        results[i] = estimator_dic

    return results

def test_model(individual_models,DataSets,create_net_func,X_test_list,y_test_list,skip_models,target):

    results = {}
    X_test = X_test_list[target]
    y_test = y_test_list[target]
    potential_sources = [c for c in DataSets if c!=target]

    # Only Network built from one source
    for source in potential_sources: 
        estimators = [individual_models[source][i] for i in individual_models[source] if i not in skip_models]
        N = create_net_func(estimators)
        results[source] = N.evaluate(X_test,y_test)
    
    # Networks built from all individual models
    estimators = [] # Networks built from all individual models
    for source in potential_sources :
        estimators.extend([individual_models[source][i] for i in individual_models[source] if i not in skip_models])
        N = create_net_func(estimators)
        results['all'] = N.evaluate(X_test,y_test)

    return results




'''
Args : results from test_all_models, list of problems to average over,  target and list of all sources
Return : res[source][model] with the average AUC over the pbs in pb_list

'''
def contract_results(results,target,DataSets):

    sources = [c for c in DataSets if c!=target]

    ##### Initialize results
    res = {}

    ###### Fill with models from 1 source only

    for source in sources: # Iterate over source sources
        res[source]={}

        res[source]['rf']=results[target][source]["Layer 0"]["Estimator 0"]
        res[source]['lr']=results[target][source]["Layer 0"]["Estimator 1"]
        res[source]['nn']=results[target][source]["Layer 0"]["Estimator 2"]
        res[source]['vote simple']=results[target][source]["Layer 1"]["Estimator 0"]
        res[source]['vote rank']=results[target][source]["Layer 1"]["Estimator 1"]
        res[source]['vote norm']=results[target][source]["Layer 1"]["Estimator 2"]
        res[source]['vote aggr']=results[target][source]["Layer 2"]["Estimator 0"]

    ###### Fill with models from all ind models

    source = 'all'
    res[source]={}

    res[source]['vote simple']=results[target][source]["Layer 1"]["Estimator 0"]
    res[source]['vote rank']=results[target][source]["Layer 1"]["Estimator 1"]
    res[source]['vote norm']=results[target][source]["Layer 1"]["Estimator 2"]
    res[source]['vote aggr']=results[target][source]["Layer 2"]["Estimator 0"]

    return res

'''
Transfrom the contracted results from contract_results to a list 
Args : contracted_results and list of sources
Return : list of scores in order described in the function
'''
def contracted_result_toList(contracted_results,sources):
    scores = []
    sources = sources+['all']
    models_ind = ['rf','lr','nn','vote simple','vote rank','vote norm','vote aggr']
    models_all = ['vote simple','vote rank','vote norm','vote aggr']

    for source in sources:
        if source=='all':
            models = models_all
        else:
            models = models_ind
        for model in models:
            scores.append(contracted_results[source][model])
            
    scores = [x[0] if isinstance(x, list) else x for x in scores ]

    return scores


'''
Plot the list of scores using ggplot to compare performance of models
'''
def ggplot_scores(scores,target,sources,problems_description):

    n_sources = len(sources)

    # Build index (the list of x_ticks)
    ind = 0
    index = []
    for i in range(n_sources):
        index.extend(range(ind,ind+7))
        ind = ind+8
    index.extend(range(ind,ind+4))

    # Axis limits
    M_y = np.max(scores)+0.025
    M_x = np.max(index)+1

    # Colors 
    c1 = "skyblue"
    c2 = "royalblue"
    c3 = "navy"
    one_source_colors = [c1]*3+[c2]*3+[c3]
    Colors = one_source_colors*n_sources+[c2]*3+[c3]

    # Text 
    title = 'Performance of different predictive models on source '+str(target)+" for "+problems_description
    xlabel = ["source "+str(x) for x in sources]+['All sources']
    xlabel = '  |  '.join(xlabel)

    # Formatting into pandas.DF
    pretty_results ={'Scores':scores,'Index':index,'Color':Colors}
    data=pd.DataFrame.from_dict(pretty_results)

    # PLotting with ggplot
    plot = ggplot(aes(x='Index',y='Scores'), data) + \
     geom_bar(stat = 'identity',fill=Colors) + \
    scale_y_continuous(limits = (0.5,M_y)) + \
    scale_x_continuous(limits = (-1,M_x)) + \
    ggtitle(title) + \
    xlab(xlabel) + ylab('ROC-AUC score') 

    return plot










