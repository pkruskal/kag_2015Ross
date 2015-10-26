import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import datetime as dt
from sklearn import tree
from sklearn import ensemble
import sklearn.grid_search as gridSearch
import time

# Change catigorical variables to numberical ones
def enumerateCatigoricals(storeSetDataFrame):

    def setHolliday2Number(hollidayTag):
        # enumerate holidays
        holidayEnumeration = {'0':1,
                              'a':2,
                              'b':3,
                              'c':4}
        if type(hollidayTag) == str:
            mapping = holidayEnumeration[hollidayTag]
        else:
            mapping = hollidayTag
        return mapping

    #trainSet['StateHoliday'] = [holidayEnumeration[hollidayTag] if type(hollidayTag) == str else hollidayTag for hollidayTag in trainSet['StateHoliday'] ]
    #testSet['StateHoliday'] = [holidayEnumeration[hollidayTag] if type(hollidayTag) == str else hollidayTag for hollidayTag in testSet['StateHoliday'] ]

    storeSetDataFrame['StateHoliday'] = [
        setHolliday2Number(hollidayTag) for hollidayTag in storeSetDataFrame['StateHoliday'] ]

    return storeSetDataFrame

# calculate the root mean squared percentage error
def calcPercentErrors(y,yhat):
    """

    """
    assert(len(y)==len(yhat))

    if type(y) != np.ndarray:
        y = np.array(y)

    if type(yhat) != np.ndarray:
        yhat = np.array(yhat)

    diffArray = y - yhat
    percentageErrors = (diffArray/y)**2

    return percentageErrors

def rmspeScore(percentageErrors):
    if type(percentageErrors) != np.ndarray:
        percentageErrors = np.array(percentageErrors)

    rmspe = np.sqrt(percentageErrors.sum()/len(percentageErrors))
    return rmspe

# enumerate the paramaters list
def enumerateParams():
    varyingParams = {
        'max_depth' : [3,5,7,9,11],
        'learning_rate' : [0.1,0.05,0.01]
    }
    aSearch = gridSearch.ParameterGrid(varyingParams)
    seachList = list(aSearch)
    #add in the set params
    for param in seachList:
        param ['n_estimators'] = 500
        param ['min_samples_split'] = 5
        param ['verbose'] = 0
    return seachList

# add in a collumn for a single store representing the competition distance
def addCompetition(thisStore,thisStoresData):
    '''
    to add information about competition in the store we will add another column
    that includes the distance away of the competition

    To account for periods where there is no competition, the competition distance
    is added as a "very high" number, psudo inf, well above other competition distances
    '''


    #initaize competition info a very high number to represent no competition
    competitionDistancesTrain = np.ones(thisStore.shape[0])*1000000.0

    #check if there even is any competition
    if ~np.isnan(thisStoresData['CompetitionOpenSinceYear'].values):
        #find the competition start data
        competitionStart = dt.datetime(
            year = thisStoresData['CompetitionOpenSinceYear'].values,
            month = thisStoresData['CompetitionOpenSinceMonth'].values,
            day = 1)
        if competitionStart < thisStore.index.max():
            competitionDist = thisStoresData['CompetitionDistance'].values[0]
            competitionDistancesTrain[thisStore.index >= competitionStart] = competitionDist

    thisStore['competition'] = competitionDistancesTrain
    return thisStore

#given a fit model, check the number of regression trees that should be used
def crossCheckNumEstimators(fitModel,crossValProps,crossValValues,trainProp,trainValues,plots = 0,saves=0):

    n_estimators = len(fitModel.estimators_)
    test_dev = np.empty(n_estimators)
    test_devPercErrors = []
    train_dev = np.empty(n_estimators)

    for i, pred in enumerate(fitModel.staged_predict(trainProp)):
        trainProp['SalesPredictions'] = pred
        trainJoined = pd.merge(trainProp,trainValues,left_index=True,right_index=True)
        percErrors = calcPercentErrors(trainJoined['Sales'],trainJoined['SalesPredictions'])
        train_dev[i] = rmspeScore(percErrors)

    for i, pred in enumerate(fitModel.staged_predict(crossValProps)):
        crossValProps['SalesPredictions'] = pred
        crossValidationJoined = pd.merge(crossValProps,crossValValues,left_index=True,right_index=True)
        percErrors = calcPercentErrors(crossValidationJoined['Sales'],crossValidationJoined['SalesPredictions'])
        test_devPercErrors.append(percErrors)
        test_dev[i] = rmspeScore(percErrors)

    if plots:
        plt.plot(np.arange(n_estimators)+1,train_dev)
        plt.plot(np.arange(n_estimators)+1,test_dev,'r')

    bestFit = np.min(test_dev)
    bestNumTrees = np.argmin(test_dev)
    bestPercErrors = test_devPercErrors[np.argmin(test_dev)]
    return bestNumTrees, bestFit, bestPercErrors

def crossValCheckParams(gradModel,trainSet,storeData,testPoint):

    storeBoostList = []
    storeFitList = []
    numTreesList = []
    modelFitList = []
    percentageErrorsList = []
    for storeID in set(trainSet['Store']):

        #isolate the store of interest
        thisStore = trainSet[trainSet['Store'] == storeID].copy()
        del thisStore['Customers']
        del thisStore['Store']

        #check if the store has enough data points
        if thisStore.shape[0] < abs(testPoint)+200:
            print 'not enough samples'
            storeBoostList.append(storeID)
            continue

        #isolate the extra data for this store
        thisStoresData = storeData[storeData['Store'] == storeID]

        #first add in competition information as a new column
        thisStore = addCompetition(thisStore,thisStoresData)

        '''
        competitionDistancesTrain = np.ones(thisStore.shape[0])*100000.0
        #check if there is any competition
        if ~np.isnan(thisStoresData['CompetitionOpenSinceYear'].values):
            #find the competition start data
            competitionStart = dt.datetime(
                year = thisStoresData['CompetitionOpenSinceYear'].values,
                month = thisStoresData['CompetitionOpenSinceMonth'].values,
                day = 1)
            if competitionStart < thisStore.index.max():
                competitionDist = thisStoresData['CompetitionDistance'].values[0]
                competitionDistancesTrain[thisStore.index >= competitionStart] = competitionDist
        thisStore['competition'] = competitionDistancesTrain
        '''

        #only train on data when the store is open and sales were positive (assuming these as outliers)
        thisStore = thisStore[thisStore['Open'] == 1]
        del thisStore['Open']
        thisStore = thisStore[thisStore['Sales'] > 0]
        thisStore.sort(inplace = True)

        #split the data to test and train sets for cross validation
        trainStore = thisStore[0:testPoint]
        crossValidationStore = thisStore[(testPoint+1):(testPoint+31)]

        #take out sales data as indexed data frames
        trainSalesDF = pd.DataFrame(trainStore['Sales'])
        crossValSalesDF = pd.DataFrame(crossValidationStore['Sales'])
        del trainStore['Sales']
        del crossValidationStore['Sales']

        #fit the model
        gradModel.fit(trainStore,trainSalesDF['Sales'])



        #cross validation check with rmspe and number of trees needed
        numTrees, bestFit, bestPercErrors = crossCheckNumEstimators(
            gradModel,
            crossValidationStore.copy(),
            crossValSalesDF,
            trainStore.copy(),
            trainSalesDF
        )

        storeFitList.append(bestFit)
        numTreesList.append(numTrees)
        modelFitList.append(gradModel)

        #keep track of all error
        percentageErrorsList.extend(bestPercErrors)

        print '.',

    #fullStoreDF = pd.concat(storeFitList)
    #rmspe = rmspeScore(fullStoreDF['Sales'].values,fullStoreDF['SalesPredictions'].values)
    return storeFitList, numTreesList, modelFitList, percentageErrorsList

def addOtherStoreInfo(trainSet,storeID,thisStore):
    '''
    This uses other store information to predict the current stores sales, who knows, may help

    :param trainSet:
     Raw training data
    :param storeID:
     Current store id
    :param thisStore:
     Isolated store info with same columns as trainSet
    :return: trainDataFrame:
    same size as trainSet, but Sales for stores are always the sales from the current store
    '''


    thisStore['predictingStore'] = 1

    otherStoreIDs = list(set(trainSet['Store']))
    del otherStoreIDs[otherStoreIDs.index(storeID)]

    newStoresList = [thisStore]
    for otherStorID in otherStoreIDs:
        otherStore = trainSet[trainSet['Store'] == otherStorID].copy()
        otherStore['predictingStore'] = 0
        del otherStore['Sales']
        newStoresList.append(pd.merge(otherStore,pd.DataFrame(thisStore['Sales']),left_index = True,right_index=True))

    trainDataFrame = pd.concat(newStoresList)
    return trainDataFrame


