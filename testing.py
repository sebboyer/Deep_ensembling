'''
Author : Sebastien Boyer
Date : January 2016

Test models 

'''

############# Dependencies
import sys
sys.path.append('/Users/sebboyer/Documents/ALFA/data_copy/features')
from dataFormat import *
from visua import *
from ensembling import *
from create_network import *

import numpy as np
from ggplot import *

from multiprocessing import *
from functools import partial
import copy

import pickle


###############################################################################
####################### Test models and Networks ##############################
###############################################################################

class Ensemble:
    def __init__(self,estimators,network_file):
        self.estimators=estimators
        self.network = pickle.load(open(network_file,"rb"))

    def train(self,X_train,y_train):

        self.network.layers[0].add_estimators(self.estimators)
        self.network.train(X_train,y_train)

    def eval(self,X,y):
        return self.network.evaluate(X,y)



'''
Apply network structure and return AUC on test (X_train/y_train to train potential meta-models)
'''
def train_and_test(network,estimators,X_train,y_train,X_test,y_test):
    network.layers[0].add_estimators(estimators)
    network.train(X_train,y_train)
    res = network.evaluate(X_test,y_test)
    return res

'''
Apply network structure and return AUC for a particular target Dataset for:
- meta models coming from each other sources individual-models
- a meta model coming from the concatenated models
- a meta model including all individual and concatenated models

'''
def cycle_train_and_test(individual_models,concat_models,DataSets,network,X_test_list,y_test_list,include_est,skip_models,target):

    results = {}
    
    potential_sources = [c for c in DataSets if c!=target]
    X_train_concat = np.concatenate([X_test_list[source] for source in potential_sources])
    y_train_concat = np.concatenate([y_test_list[source] for source in potential_sources])

    X_test ,y_test = X_test_list[target],y_test_list[target]

    networks = [copy.deepcopy(network) for i in range(len(potential_sources)+2)]

    ########## Only Network built from one source
    if 'unique_source' in include_est :
        for j,source in enumerate(potential_sources): 
            
            estimators = [individual_models[source][i] for i in individual_models[source] if i not in skip_models] 
            
            X_train ,y_train = X_test_list[source],y_test_list[source]
            results[source] = train_and_test(networks[j],estimators,X_train,y_train,X_test,y_test)
        
    ########## Networks built from concatenated data
    if 'concat' in include_est:
        estimators = [concat_models[target][i] for i in concat_models[target] if i not in skip_models]
        results['concat'] = train_and_test(networks[-1],estimators,X_train_concat,y_train_concat,X_test,y_test)
    
    ########## Networks built from all individual models
    if 'all' in include_est:
        estimators = [] 
        for source in potential_sources :
            estimators.extend([individual_models[source][i] for i in individual_models[source] if i not in skip_models])
        results['all'] = train_and_test(networks[-2],estimators,X_train_concat,y_train_concat,X_test,y_test)

    return results

'''
Apply cycle_train_and_test for each possible target as defined in DataSets
return dictionnary of results 
results[target][source]
'''

def test_models_pb(individual_models,concat_models,DataSets,X_test_list,y_test_list,X_train_list,y_train_list,folder,network_name,include_est,skip_models=[]):
    
    results ={}
    
    for target in DataSets:
        print "# Target = ",target
        
        X_test = X_test_list[target]
        y_test = y_test_list[target]
        
        network = load_net(folder,network_name)
        result_target = cycle_train_and_test(individual_models,concat_models,DataSets,network,X_test_list,y_test_list,include_est,skip_models,target)
        
        results[target] = result_target
        
    return results





'''
Test all models (All pb + networks + target) 

'''

def test_all_models(individual_models,feat_dic_list,featids_feat,pb_list,structure_folder,networks,concat_folder,include_est=['all'],para=0,skip_models=[],DataSets=[]):

    if DataSets == []:
        DataSets = range(len(feat_dic_list))
    
    results = {}
    
    def test_models_pb_aux(individual_models,feat_dic_list,featids_feat,structure_folder,networks,concat_folder,include_est,pb):
        
        print "########### Problem = ",pb

        feat_weekids=[pb[1]]
        label_weekid=pb[0]
        
        train_ratio=0.7
        X_train_list,X_test_list,y_train_list,y_test_list=prepare_problem_data_all(feat_dic_list,featids_feat,feat_weekids,label_weekid,train_ratio)
        
        individual_models_pb = individual_models[pb]
        concat_models_pb = pickle.load(open(concat_folder+"concat_model_("+str(pb[0])+", "+str(pb[1])+").p","rb"))
        
        result_pb = {}
        for network_name in networks:
            print "###### Testing network : "+network_name
            result_pb[network_name] = test_models_pb(individual_models_pb,concat_models_pb,DataSets,X_test_list,y_test_list,X_train_list,y_train_list,structure_folder,network_name,include_est,skip_models)
            
        return result_pb
    
    
    ##### Fill results
    if para==1:
        partial_test_models_pb = partial(test_models_pb,individual_models,feat_dic_list,featids_feat,structure_folder,networks,concat_folder,pb)
        pool = Pool()
        results_dic = pool.map(partial_test_models_pb,pb_list)
    else:
        results_list = []
        for pb in pb_list:
            results_list.append(test_models_pb_aux(individual_models,feat_dic_list,featids_feat,structure_folder,networks,concat_folder,include_est,pb))
        # Format results into dic
        results = {}
        for i,estimator_dic in enumerate(results_list):
            results[pb_list[i]] = estimator_dic
    
    return results

    


# '''
# Transform a results dictionnary of the form 
# results[pb][target][source][layer][estimator] (dictionnary)
# into a set of 3 lists :
# - scores 
# - indexes
# - colors
# for display and comparison purposes
# '''
# def transform_results(results,pb,target):

#     c1 = "skyblue"
#     c2 = "royalblue"
#     c3 = "navy"
#     c4 = 'black'

#     colors_list = [c1,c1,c1,c1,c2]

#     scores,index,colors = [],[],[]

#     ind = 0

#     results_exp = results['1L_simple'][target]
#     #### Add All models AUC
#     for source in results_exp:
#         if source != 'all': # For simple source
#             color_ind = 0
#             for layer in results_exp[source]:
#                 for est in results_exp[source][layer]:
#                     scores.append(results_exp[source][layer][est])
#                     index.append(ind)
#                     colors.append(colors_list[color_ind])
#                     ind+=1
#                     color_ind+=1
#             ind+=1
            
#     #### Add Network Full AUC
#     ind+=1
#     scores.append(results_exp['all']['Layer 1']['Estimator 0'])
#     index.append(ind)
#     colors.append(c3)

#     #### Add Network '1L_LR'
#     results_exp = results_networks['1L_LR'][target]
#     ind+=2
#     scores.append(results_exp['all']['Layer 1']['Estimator 0'])
#     index.append(ind)
#     colors.append(c3) 

#     #### Add Network '2L_LR'
#     results_exp = results_networks['2L_simple'][target]
#     ind+=2
#     scores.append(results_exp['all']['Layer 2']['Estimator 0'])
#     index.append(ind)
#     colors.append(c3)   

#     #### Add Network '2L_LR'
#     results_exp = results_networks['2L_LR'][target]
#     ind+=2
#     scores.append(results_exp['all']['Layer 2']['Estimator 0'])
#     index.append(ind)
#     colors.append(c3)  
      
#     ## Formatting into pandas.DF
#     pretty_results ={'Scores':scores,'Index':index,'Color':colors}
#     data=pd.DataFrame.from_dict(pretty_results)

#     title = 'Performance of Classifiers'
#     xlabel = 'Light blue : simple classifiers  |  Medium blue : Vote inside one source  |  Navy blue : Ensembling networks'

#     # Axis limits
#     M_y = np.max(scores)+0.025
#     M_x = np.max(index)+1

#     # PLotting with ggplot
#     plot = ggplot(aes(x='Index',y='Scores'), data) + \
#      geom_bar(stat = 'identity',fill=colors) + \
#     scale_y_continuous(limits = (0.5,M_y)) + \
#     scale_x_continuous(limits = (-1,M_x)) + \
#     ggtitle(title) + \
#     xlab(xlabel) + ylab('ROC-AUC score')

#     return plot












