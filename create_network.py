'''

Author : Sebastien Boyer
Date : Januray 2016

'''


import numpy as np
from ensembling import *
import pickle
import networkx as nx
import matplotlib.pyplot as plt





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


### Visualization network

def plot_graph(net,results=0,savefile=0):
    
    # Initialize graph
    pos = {}
    nodes = {} # layer : node_list
    labels = {}
    node =0
    max_nodes = max([len(layer.estimators) for layer in net.layers])
    n_layers = len(net.layers)
    G=nx.Graph()
    
    if results==0:
        res=0
    else:
        res=1
        
    if savefile==0:
        save=0
    else:
        save =1
        
    plot = plt.figure(figsize=(10,10))
    
    
    for i,layer in enumerate(net.layers):
#         print i
        nodes[i]= []
        n = len(layer.estimators)

        ######### Nodes
        for j,est in enumerate(layer.estimators):
#             print j
            nodes[i].append(node) 
            #Label
            if res:
                labels[node] = "%.2f" % (results['Layer '+str(i)]['Estimator '+str(j)])
            else:
                if i>0:
                    m = layer.models[j]
                    if not isinstance(m,Model):
                        labels[node] = m.vote_type
                    else:
                        labels[node] = m.class_type
                else:
                    labels[node] = j
            #Add  
            G.add_node(node)

            # Pos
            if n%2==0:
                y_pos = (j+0.5)/float(n)
            else:
                y_pos = (j+0.5)/float(n)
            pos[node] = [i,y_pos]
            node+=1

        ########### Draw edges
        if i!=0:
            if layer.links == {}:
                for child in nodes[i]:
                    for parent in nodes[i-1]:
                        G.add_edge(parent,child)
            else:
                for parent_id in layer.links:
                    for child_id in layer.links[parent_id]:
                        G.add_edge(nodes[i-1][parent_id],nodes[i][child_id])
                
                
    ## Draw graph
    nx.draw_networkx_labels(G,pos,labels,font_size=20)
    nx.draw_networkx_nodes(G,pos,
                           node_color='gray',
                           node_size=2800,
                       alpha=0.7)        
    nx.draw_networkx_edges(G,pos,
                           width=8,alpha=0.2,edge_color='gray')
    plt.axis('off')
    
    if save:
        plt.savefig(savefile)
    
#     return plot



