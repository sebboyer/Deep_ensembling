# Deep Ensembling

Work in progress !

Deep_ensembing is a package that let you train models on different data-sets and combine their prediction to build a robust predictive model. This package is intended to prediction problems meeting the following criteria :
- Several full data sources (both features and outcomes available) available
- A binary classification problem

This package let you define easily what kind of classification model to build and how to combine them. The outcome of the workflow is a robust predictive model that is likely to perform very well on a new data source.

![An example of a simpel Ensembling structure](Pictures/why.jpg)

## How to use : Train models

The script training.py allows you to train multiple binary models on multiple training Datasets in parallel.
The following lines will train a LogisticRegression, a NearestNeighbor and a RandomForest classifier on the list of dataset provided in X_train_list and y_train_list (each of the data in the list X_train_list must share the same second dimension) :

```python
import training as train

model_list = ['lr','rf','nn']
params_list = [(1,4),(0,0),(80,150)]
output_filename = 'TrainedModels.p'
train.main(X_train_list,y_train_list,model_list,params_list,output_filename=output_filename)
```
This will pickle the trained models in output_filename.

![](Pictures/structure2.jpg)

Options include :
- para : if para=1 run in parallel 
- model_type : if model_type='concat' trained concatenated models (concatenate all dataSet but one and train models on this concatenation, do it for each hold-out DataSet).

## How to use : Build Ensembling structure

This package allows you to create from very simple to very complex "Ensembling method" always with very little code. 

A structure is defined as an object from class 'Network'. When no 'links' between layers are mentionned the fully-connected option is default. We show the code to create the two structures shown in Figure 2.

```python
from ensembling import *

#### First Structure 
N=Network() # Define Network
N.add_layer("Models_layer",[]) # First layer
N.add_layer("Output_layer",[Vote("simple")])  # Last Layer

#### Second Structure 
N=Network() # Define Network
N.add_layer("Models_layer",[]) # First layer

n_input_models_per_source = 3 # Second layer
n_sources = 3
n_output_models_per_source = 2
N.add_layer("Hidden_layer",[Vote("simple"),Vote("norm")])
N.layers[1].links = create_independent_links(n_input_models_per_source,n_sources,n_output_models_per_source)

N.add_layer("Hidden_layer",[Vote("simple"),Vote("rank"),Vote("norm")]) # Third Layer
N.add_layer("Output_layer",[Vote("simple")])  # Fourth Layer
```
![](Pictures/examples.jpg)

A default structure is provided, use :

```python
from ensembling import *

N = create_simple_network()
```


## How to use : Test models

The script testing.py allows you to test you trained models. More importantly it allows you to have them vote in a structured way that you can define. We provide a default structure :
- create_net_func = create_simple_network : is a structure where all provided models vote in three different ways (classic sum-vote, rank-based vote, and normalized vote), then those three votes are aggregated using a last classic vote to produce the final output.

The following lines will test the AUC of all pre-trained models in 'TrainedModels.p' on the list of DataSet provided in X_test_list, y_test_list (second dimension of DataSets in X_test_list must be the same as the second dimensions of the DataSets used during training).

```python
import testing as test
import pickle

models = pickle.load(open('TrainedModels.p','rb'))
results = test.test_all_models(models,X_test_list,y_test_list)
```

Options inlcude :
- skip_models : list of model names to skip (among 'lr', 'nn', 'rf', 'svm')
- para : if para=1 run test in parallel
- DataSets : list of source index to take into account (default is all)
