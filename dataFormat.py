'''
Author Sebastien Boyer
Date : Oct 2015

Script formatting data from feature csv for predictive tasks

'''


import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import copy
from sklearn.cross_validation import train_test_split



# Read the csv (filename) containing the features extracted for one course 
# Returns an array containing the rows an columns of the csv file
def read_csv_features(filename):
    data=[]
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            data.append(row)
    data=np.array(data)
    return data


# Read an array containing the features
# Return a dictionnary containing all the feature values structured as follows : feat_dic[userid][weekid][feat_id]=feat_value
def dic_from_data(data):
    feat_dic={}
    featids=[]
    count_duplicate={}
    for row in data:
        userid=row[2]
        featid=int(row[1])
        weekid=int(row[3])
        value=float(row[4])

        if featid not in featids:
            featids.append(featid)

        if userid not in feat_dic:
            feat_dic[userid]={}

        if weekid not in feat_dic[userid]:
            feat_dic[userid][weekid]={}

        if featid in feat_dic[userid][weekid]:
            if featid not in count_duplicate:
                count_duplicate[feat_id]=0
            count_duplicate[feat_id]+=1

        feat_dic[userid][weekid][featid]=value
    return feat_dic,featids,count_duplicate

# Take the feature dictionnary of one course, the weeks to use as features and the week to use as label
# Returns the X matrix containing features, and the y column containing labels for that particular problem
def create_feat_mat(feat_dic,feat_weekids,featids,label_weekid):
    X=[]
    y=[]
    
    for userid in feat_dic:
        userid_feat=[]
        
        # Don't count userid if we know he has already dropout
        if feat_dic[userid][max(feat_weekids)][1]==0:
            continue
            
        for weekid in feat_weekids:
            userid_feat.extend([feat_dic[userid][weekid][x] if x in feat_dic[userid][weekid] and x!=1 else 0 for x in featids ])
        
        X.append(userid_feat)
        y.append(feat_dic[userid][label_weekid][1])
    return np.array(X),np.array([y]).T

# Normalize the list of matrix in X_list with the same transformation
def normalize_jointly(X_list):   
    concat=np.concatenate(X_list,axis=0)

    m=np.min(concat,axis=0)
    M=np.max(concat,axis=0)
    mean=np.mean(concat,axis=0)

    M[M==m]=1
    for i,X in enumerate(X_list):
        X_list[i]=(X-mean)/(M-m)
    return X_list

# Take a concatenate matrix whose last column is the label and split into X_train,X_test,y_train,y_test
def split(full,train_ratio):
    train,test=train_test_split(full,train_size=train_ratio,random_state=1234)
    
    X_train=train[:,:-1]
    y_train=train[:,-1]

    X_test=test[:,:-1]
    y_test=test[:,-1]

    return X_train,X_test,y_train,y_test

# Read list of files and create list of dictionnary, the set of common features
def extract_structured_data_from_file(files):

    ## Extracting data from file
    n=len(files)
    data_list=[]
    for f in files:
        data_list.append(read_csv_features(f))

    ##Transforming data format 
    feat_dic_list=[]
    featids_list=[]
    count_duplicate_list=[]
    for d in data_list:
        feat_dic,featids,count_duplicate=dic_from_data(d)
        feat_dic_list.append(feat_dic)
        featids_list.append(featids)
        count_duplicate_list.append(count_duplicate)

    ## Compute intersection feature list
    inter=featids_list[0]
    for i in range(n):
        set(inter).intersection(featids_list[i])
    featids_feat=inter

    return feat_dic_list,featids_feat,count_duplicate_list

# Read feature dictionnary and other
# Return useable splitted X_train,X_test,y_train,y_test
def prepare_problem_data(feat_dic1,featids_feat,feat_weekids,label_weekid,train_ratio):
    X1,y1=create_feat_mat(feat_dic1,feat_weekids,featids_feat,label_weekid)
    full1=np.concatenate((X1, y1), axis=1)
    X_train1,X_test1,y_train1,y_test1=split(full1,train_ratio)
    [X_train1,X_test1]=normalize_jointly([X_train1,X_test1])
    return X_train1,X_test1,y_train1,y_test1

# Apply prepare_problem_data to all courses
def prepare_problem_data_all(feat_dic_list,featids_feat,feat_weekids,label_weekid,train_ratio):
    n=len(feat_dic_list)
    X_train_list=[]
    X_test_list=[]
    y_train_list=[]
    y_test_list=[]
    for i in range(n):
        X_train,X_test,y_train,y_test=prepare_problem_data(feat_dic_list[i],featids_feat,feat_weekids,label_weekid,train_ratio)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return X_train_list,X_test_list,y_train_list,y_test_list










