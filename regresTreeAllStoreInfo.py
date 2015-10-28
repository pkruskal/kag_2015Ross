import standardDefs as ross
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import datetime as dt
from sklearn import tree
from sklearn import ensemble
import sklearn.grid_search as gridSearch
import time

# enumerate the paramaters list
def enumerateParams():
    varyingParams = {
        'max_depth' : [3,4,5,6,7,8,9,19,11],
        'learning_rate' : [0.8,0.5,0.3,0.2,0.1,0.05,0.01]
    }
    aSearch = gridSearch.ParameterGrid(varyingParams)
    seachList = list(aSearch)
    #add in the set params
    for param in seachList:
        param ['n_estimators'] = 500
        param ['min_samples_split'] = 5
        param ['verbose'] = 0
    return seachList


trainSet = pd.read_csv('../data/train.csv',index_col='Date',parse_dates=True)
storeData =  pd.read_csv('../data/store.csv')

trainSet = ross.enumerateCatigoricals(trainSet)

