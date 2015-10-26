__author__ = 'peter'

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import datetime as dt
from sklearn import tree
from sklearn import ensemble
import sklearn.grid_search as gridSearch
import time
import petersStandard as pk


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



trainSet = pd.read_csv('../data/train.csv',index_col='Date',parse_dates=True)
testSet = pd.read_csv('../data/test.csv',index_col='Date',parse_dates=True)

storeData =  pd.read_csv('../data/store.csv')

trainSet = enumerateCatigoricals(trainSet)
testSet = enumerateCatigoricals(testSet)

testPoint = np.random.randint(-100,-32)


testPoint = -32
iBoot = 1

searchList = enumerateParams()

storePramDictList = []
collapsedParamDictList = []
time.clock()
for iParam, params in enumerate(searchList):
    tic = time.clock()

    print 'Testing paramset num ' + str(iParam) + ' out of ' + str(len(searchList))

    gradModel = ensemble.GradientBoostingRegressor(**params)

    storeFitList, numTreesList, modelFitList, percentageErrorsList = \
        crossValCheckParams(gradModel,trainSet,storeData,testPoint)

    #save modelFitList, numTreesList, storeFitList

    for iStore, storeID in enumerate(set(trainSet['Store'])):
        storeParams = params.copy()
        storeParams['StoreID'] = storeID
        storeParams['numTrees'] = numTreesList[iStore]
        storeParams['fit'] =  storeFitList[iStore]
        storeParams['bootID'] = iBoot

        storePramDictList.append(storeParams)

    params['bestFit'] = rmspeScore(percentageErrorsList)
    collapsedParamDictList.append(params)

    print params
    print 'time check ' + str((time.clock()-tic)/60.0)




combinedParams = pd.DataFrame(collapsedParamDictList).sort('bestFit')
storeParams = pd.DataFrame(storePramDictList)

#bestFits = storeParams.groupby(['StoreID']).max()

thisStore = storeParams[storeParams['StoreID'] == 14].sort('fit')
thisStore.ix[thisStore.index[0]]
thisStore.head(5)
plt.hist(thisStore['fit'].values)

bestStoreFits
for storeID in set(trainSet['Store']):
    thisStore = storeParams[storeParams['StoreID'] == 14].sort('fit')
    thisStore.ix[thisStore.index[0]]


#######################
#   paramater search  #
#######################


'''
depth = 15
#default tree
treeModelTest = tree.DecisionTreeRegressor(max_depth=depth)

# restrict to month for traiing and testing
testPoint = np.random.randint(-100,-28)


storeBoostList = []
storeID = 1115
for storeID in storIDs:
    thisStore = trainSet[trainSet['Store'] == storeID]
    thisStore = thisStore[thisStore['Open'] == 1]
    thisStore.sort(inplace = True)

    storeSales = thisStore['Sales']

    del thisStore['Customers']
    del thisStore['Sales']
    del thisStore['Store']

    if storeSales.shape[0] < abs(testPoint)+100:
        print 'not enough samples'
        storeBoostList.append(storeID)
        continue

    trainSales = storeSales[0:testPoint]
    trainStoreProps = thisStore[0:testPoint]
    crossValidationStoreProps = thisStore[(testPoint+1):(testPoint+30)]
    crossValidationSales = storeSales[(testPoint+1):(testPoint+30)]

    #default fit
    treeModelTest.fit(trainStoreProps,trainSales)
    pred = treeModelTest.predict(crossValidationStoreProps)

    crossValidationSales.values-pred
    plt.figure()
    trainSales.plot()
    plt.plot(crossValidationSales)
    crossValidationSales.plot()
    prectedSales = crossValidationSales.copy()
    prectedSales[:] = pred
    prectedSales.plot()



for maxDepth in np.arange(3,13)
    gradModel = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=3,verbose=0)
    for iboot in np.arange(5):
        testPoint = np.random.randint(-100,-28)
        treeNums, estErrors = crossValCheck(gradModel,trainSet,testPoint)

'''




'''
gridSearch._fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score=False, return_parameters=False, error_score='raise')
looks like it does cross validation for you but maybe not?

just making the paramater search simple
params = {'a' : [1,4,6],'b' : ['a','g','s'],'c' : [True,False]}
aSearch = gridSearch.ParameterGrid(params)
seachList = list(aSearch)

Note multi processing tool
pool = multiprocessing.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
'''






###################################
#   assume have good paramaters   #
###################################


#default tree
depth = 15
treeModelTest = tree.DecisionTreeRegressor(max_depth=depth)

ensambleTreeModel = ensemble.GradientBoostingRegressor(n_estimators=10, max_depth=3,verbose=0)

storeBoostList = []
storeFitList = []
storeFitList2 = []
for storeID in set(testSet['Store']):
    thisStore = trainSet[trainSet['Store'] == storeID].copy()
    testStore = testSet[testSet['Store'] == storeID].copy()

    competitionDistancesTrain = np.ones(thisStore.shape[0])*100000.0
    competitionDistancesTest = np.ones(testStore.shape[0])*100000.0
    if ~np.isnan(storeData[storeData['Store']==storeID]['CompetitionOpenSinceYear'].values):
        competitionStart = dt.datetime(storeData[storeData['Store']==storeID]['CompetitionOpenSinceYear'].values,storeData[storeData['Store']==storeID]['CompetitionOpenSinceMonth'].values,1)
        if competitionStart < thisStore.index.max():
            competitionDist = storeData[storeData['Store'] == storeID]['CompetitionDistance'].values[0]
            competitionDistancesTrain[thisStore.index >= competitionStart] = competitionDist
            competitionDistancesTest = np.ones(testStore.shape[0])*competitionDist

    thisStore['competition'] = competitionDistancesTrain
    testStore['competition'] = competitionDistancesTest

    thisStore = thisStore[thisStore['Open'] == 1]
    thisStore.sort(inplace = True)
    storeSales = thisStore['Sales']

    storesTestID = pd.DataFrame(testStore['Id'])
    testStore.sort(inplace = True)

    testStore2 = testStore.reset_index(level = 1)
    testStore2.set_index('Id',inplace = True)
    del testStore2['Date']
    del testStore2['Store']

    del thisStore['Customers']
    del thisStore['Sales']
    del thisStore['Store']
    del testStore['Store']
    del testStore['Id']

    if storeSales.shape[0] < 300:
        print 'not enough samples'
        storeBoostList.append(storeID)
        continue

    #default fit
    treeModelTest.fit(thisStore,storeSales)
    predSales = treeModelTest.predict(testStore[testStore['Open'] == 1])
    predSales2 = treeModelTest.predict(testStore2[testStore2['Open'] == 1])

    fullPredSales = np.zeros(storesTestID.shape[0])
    fullPredSales[np.where(testStore['Open'] == 1)] = predSales
    storesTestID['Sales'] = fullPredSales
    storeFitList.append(storesTestID)

    testStore2['Sales'] = 0
    testStore2.loc[testStore2['Open'] == 1,'Sales'] = predSales2
    testStore2 = testStore2.reset_index(level = 1)

    storeFitList2.append(testStore2[['Id','Sales']])

outputDF = pd.concat(storeFitList2)


outputDF = outputDF.reset_index(1)
outputDF = outputDF.set_index('Id')
del outputDF['index']
outputDF.sort(inplace = True)
outputDF.to_csv('D:/rossmann/petersTestEnsambleTree2.csv')
