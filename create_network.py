'''

Author : Sebastien Boyer
Date : Januray 2016

'''


import numpy as np
from ensembling import *
import pickle





###############################################################################
########################## Network Creation ###################################
###############################################################################
'''
Return a simple 3 Layers network build from the given estimators
'''

def create_simple_network():
    
    ### Define Network
    N=Network()
    # First layer
    N.add_layer("Models_layer",[])
    # Second Layer
    N.add_layer("Hidden_layer",[Vote("simple"),Vote("rank"),Vote("norm")])
    # Third Layer
    N.add_layer("Output_layer",[Vote("simple")])

    # Train Network (nothing here because only Vote models)
    N.train()
    
    return N

'''
Create the links needed to link to layers in an bucket-fashion (all models from the same sources are aggregated but not models across sources)
Return the links ready to be set as argument to the layer

'''
def create_independent_links(n_estimators_per_source,n_sources,n_models_per_source):
    links ={}
    
    course_estimators = {}
    for s in range(n_sources):
        course_estimators[s]=range(s*n_estimators_per_source,(s+1)*n_estimators_per_source)
        
    for s in range(n_sources):    
        for m in range(n_models_per_source):
            links[s*n_models_per_source+m]=course_estimators[s]
            
    return links


'''
Dump a network structure to a .p file
'''
def create_net(network,folder,name):
    pickle.dump(network,open(folder+"Network_"+name+'.p','wb'))

'''
Load a network structure from a .p file
'''
def load_net(folder,name):
    return pickle.load(open(folder+"Network_"+name+'.p',"rb"))

'''
Add estimators to Network
'''
def add_estimators(N,estimators):
    N.layers[0].add_estimators(estimators)



