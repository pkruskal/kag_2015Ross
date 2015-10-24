import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import datetime as dt
import re
import scipy.signal as sig

trainSet = pd.read_csv('../data/train.csv',index_col='Date',parse_dates=True)
testSet = pd.read_csv('../data/test.csv',index_col='Date',parse_dates=True)

storeData =  pd.read_csv('../data/store.csv')

#print an inital report of the data
print 'training starts at ' + str(trainSet.index.min())
print 'training ends at ' + str(trainSet.index.max())
print 'predicting starts at ' + str(testSet.index.min())
print 'predicting ends at ' + str(testSet.index.max())

#extract stores
print 'there are ' + str(len(set(trainSet['Store']))) + ' unique stores in the data set'

print 'the general structure and stats of the data '
print trainSet.describe()

#some initial boxplots
trainSet.boxplot('Sales','DayOfWeek')
trainSet.boxplot('Sales','Promo')
trainSet.boxplot('Sales','StateHoliday')
trainSet.boxplot('Sales','SchoolHoliday')


#create a data frame for a single store and plot timeseres
def plotStoresTimeSeries(trainSet,storeData,storeID,savepath = None):
    #storeID = 3
    thisStore = trainSet[trainSet['Store'] == storeID]

    #set date as the index
    #thisStore = thisStore.set_index('Date')

    plt.subplot(3,1,1)
    plt.title('store ' + str(storeID) + ' competitionDistance ' + str(storeData[storeData['Store'] == storeID]['CompetitionDistance'].values))
    storeOpen = thisStore[thisStore['Open'] == 1]
    storeOpen['Customers'].plot()
    if ~np.isnan(storeData[storeData['Store']==storeID]['CompetitionOpenSinceYear'].values):
        competitionStart = dt.datetime(storeData[storeData['Store']==storeID]['CompetitionOpenSinceYear'].values,storeData[storeData['Store']==storeID]['CompetitionOpenSinceMonth'].values,1)
        if competitionStart < thisStore.index.max():
            storeOpen[storeOpen.index >= competitionStart]['Customers'].plot()
    plt.ylabel('Customers')

    plt.subplot(3,1,2)
    storeOpen['Sales'].plot()
    if storeData[storeData['Store']==storeID]['Promo2'].values == 1:
        promoStart = dt.datetime.fromordinal(dt.datetime(storeData[storeData['Store']==storeID]['Promo2SinceYear'].values,1,1).toordinal() + 7*storeData[storeData['Store']==storeID]['Promo2SinceWeek'].values)
        storeOpen[storeOpen.index >= promoStart]['Sales'].plot()
    plt.ylabel('Sales')

    plt.subplot(3,1,3)
    thisStore.ix[thisStore['Promo'] == 1,'Promo'] = 1
    thisStore[(thisStore['Open'] == 1)]['Promo'].plot(style='o')

    thisStore.ix[thisStore['SchoolHoliday'] == 1,'SchoolHoliday'] = 2
    thisStore[(thisStore['Open'] == 1)]['SchoolHoliday'].plot(style='o')

    thisStore.ix[thisStore['StateHoliday'] == '0','StateHoliday'] = 0
    thisStore.ix[thisStore['StateHoliday'] == 'a','StateHoliday'] = 3
    thisStore.ix[thisStore['StateHoliday'] == 'b','StateHoliday'] = 4
    thisStore.ix[thisStore['StateHoliday'] == 'c','StateHoliday'] = 5
    theseholidays = thisStore[(thisStore['Open'] == 1)]['StateHoliday']
    theseholidays.plot(style='o')
    plt.ylim([0.5,5.5])
    plt.ylabel('events')
    if savepath:
        plt.savefig(savepath,format='jpg')
        plt.clf()


def plotStoreDailyTrends(trainSet,storeData,storeID,savepath = None):

    thisStore = trainSet[trainSet['Store'] == storeID]
    thisStore = thisStore[thisStore['Open'] == 1]

    plt.figure()
    plt.violinplot(
        [thisStore[thisStore['DayOfWeek'] == dow]['Sales'] for dow in set(thisStore['DayOfWeek'])],
        showmeans=True)
    plt.boxplot(
        [thisStore[thisStore['DayOfWeek'] == dow]['Sales'] for dow in set(thisStore['DayOfWeek'])],
        notch=1)
    plt.xlabel('day of week')
    plt.ylabel('sales')


    storeCompetitionFlag = ~np.isnan(storeData[storeData['Store']==storeID]['CompetitionOpenSinceYear'].values)

    if storeCompetitionFlag:
        thisStore = thisStore

    #restrict to time after competition if this includes enough time, if not maybe consider other stats



#as regression tree need to define the scorer creativly

# base line takes days into account
# can take promotions into account
# can take hollidays into account
# can take competition into account
# can take stor type, and or assortment type into account






def testCompetition():
    #to look at the changes in dail stats due to competition
    #maybe establish some thresholds





#plot time series for all the stores
plt.figure(figsize=[20,9])
for storeNum in set(trainSet['Store']):
    if storeNum > 688:

        storeType = storeData[storeData['Store'] == storeNum]['StoreType'].values[0]
        #a,b,c,d
        storeAssortment = storeData[storeData['Store'] == storeNum]['Assortment'].values[0]
        #a,b,c

        print str(storeNum) + ' type ' + storeType + ' assortment ' + storeAssortment

        savePath = '../figures/storeTimeseries/' + \
                   'type_' + storeType + '_assortment_' + storeAssortment + \
                   '/competition' + \
                   str(storeData[storeData['Store'] == storeNum]['CompetitionDistance'].values[0].astype(int)) + \
                   '_store' + str(storeNum) + '.jpg'
        plotStoresTimeSeries(trainSet,storeData,storeNum,savePath)




####
# tests on a single store 2015-10-21
####
storeNum = 1108
thisStore = trainSet[trainSet['Store'] == 1108]
thisStore = thisStore[thisStore['Open']==1]
storeType = storeData[storeData['Store'] == storeNum]['StoreType'].values[0]
storeAssortment = storeData[storeData['Store'] == storeNum]['Assortment'].values[0]



plt.figure()
plt.violinplot(
    [thisStore[thisStore['DayOfWeek'] == dow]['Sales'] for dow in set(thisStore['DayOfWeek'])],
    showmeans=True)
plt.boxplot(
    [thisStore[thisStore['DayOfWeek'] == dow]['Sales'] for dow in set(thisStore['DayOfWeek'])],
    notch=1)
plt.xlabel('day of week')
plt.ylabel('sales')


promotedStore = thisStore[thisStore['Promo'] == 1]
unpromotedStore = thisStore[thisStore['Promo'] == 0]
plt.figure()
plt.violinplot(
    [promotedStore[promotedStore['DayOfWeek'] == dow]['Sales'] for dow in set(promotedStore['DayOfWeek'])],
    showmeans=True)
plt.violinplot(
    [unpromotedStore[unpromotedStore['DayOfWeek'] == dow]['Sales'] for dow in set(unpromotedStore['DayOfWeek'])],
    showmeans=True)
plt.boxplot(
    [promotedStore[promotedStore['DayOfWeek'] == dow]['Sales'] for dow in set(promotedStore['DayOfWeek'])],
    notch=1)
plt.boxplot(
    [unpromotedStore[unpromotedStore['DayOfWeek'] == dow]['Sales'] for dow in set(unpromotedStore['DayOfWeek'])],
    notch=1)
plt.xlabel('day of week')
plt.ylabel('sales')










idx = pd.date_range(dt.datetime(2013,1,1,00,00,00),dt.datetime(2015,7,31,00,00,00),freq = 'D')
salesMatDay1 = np.zeros([len(set(trainSet['Store'])), len(idx)])
for irow, storeID in enumerate(set(trainSet['Store'])):
    thisStore = trainSet[trainSet['Store'] == storeID]
    theseSales = thisStore[thisStore['DayOfWeek'] == 1].Sales
    theseSales = theseSales.reindex(idx)
    salesMatDay1[irow,:] = theseSales.values

dowpd = thisStore.groupby(thisStore['DayOfWeek']).count()
dowpd = thisStore.groupby(thisStore['DayOfWeek']).std()

thisStore[thisStore['DayOfWeek'] == 3]['Sales'].values[:134]-thisStore[thisStore['DayOfWeek'] == 6]['Sales'].values[:134]

storeID = 2

def plotStoresScatterByIndicator(trainSet):
    storeID = 1
    thisStore = trainSet[trainSet['Store'] == storeID]
    plt.figure(figsize=[15,15])
    plt.plot(thisStore[thisStore['Open'] == 1]['Customers'],thisStore[thisStore['Open'] == 1]['Sales'],'k.')
    plt.plot(thisStore[(thisStore['Promo'] == 1) * (thisStore['Open'] == 1)]['Customers'],
             thisStore[(thisStore['Promo'] == 1) * (thisStore['Open'] == 1)]['Sales'],'rs')
    plt.plot(thisStore[(thisStore['StateHoliday'] == 1) * (thisStore['Open'] == 1) ]['Customers'],
             thisStore[(thisStore['StateHoliday'] == 1) * (thisStore['Open'] == 1)]['Sales'],'go')
    plt.plot(thisStore[(thisStore['SchoolHoliday'] == 1) * (thisStore['Open'] == 1)]['Customers'],
        thisStore[(thisStore['SchoolHoliday'] == 1)* (thisStore['Open'] == 1)]['Sales'],'bd')
    plt.xlabel('customers')
    plt.xlabel('sales')
    plt.legend(['no indicator','promotion','state holliday','school holiday'])
    plt.title('customers and sales by indicator')
    plt.savefig('../figures/storeScatters/byIndicator' + str(storeID))

def plotStoresScatterByDay():
    storeID = 1
    thisStore = trainSet[trainSet['Store'] == storeID]

    plt.figure(figsize=[30,10])

    for day in np.arange(1,8):
        plt.plot(thisStore[(thisStore['DayOfWeek'] == day) * (thisStore['Open'] == 1)]['Customers'],
                 thisStore[(thisStore['DayOfWeek'] == day) * (thisStore['Open'] == 1)]['Sales'],'.')


    plt.xlabel('customers')
    plt.xlabel('sales')

    plt.legend(['Day' + str(day) for day in np.arange(1,8)])
    plt.title('customers and sales by indicator')
    plt.savefig('../figures/storeScatters/byDay' + str(storeID))


grouptedRateMedian = trainSet.groupby(trainSet.Store).median()
plt.hist(grouptedRateMedian['Sales'].values,100)
grouptedRateMean = trainSet.groupby(trainSet.Store).mean()
plt.hist(grouptedRateMean['Sales'].values,100)
grouptedRateStd = trainSet.groupby(trainSet.Store).std()
plt.hist(grouptedRateMean['Sales'].values,100)
groupRatedCount = trainSet.groupby(trainSet.Store).count()
grouptedRateMean['ste'] = grouptedRateStd['Sales']/np.sqrt(groupRatedCount['Sales'])


class Store(object):

    def __init__(self,fullDataFrame,storeData,storeIndx):
        self.storeIndx = storeIndx

        self.data = fullDataFrame[fullDataFrame['Store'] == storeIndx]

        '''
        sales, customers, openFlag, promo, stateHoliday, schoolHoliday, dayOfWeek, timeStamps
        '''

        self.daysFrom2014 = [dt.datetime.toordinal(tstamp)-dt.datetime.toordinal(dt.datetime(2014,1,1)) for tstamp in fullDataFrame[fullDataFrame['Store'] == storeIndx].index]

        self.storeType = storeData[storeData['Store']==storeIndx]['StoreType']
        self.assortment = storeData[storeData['Store']==storeIndx]['Assortment']

        self.promo2Flag = storeData[storeData['Store']==storeIndx]['Promo2']
        self.promoStartWeek = storeData[storeData['Store']==storeIndx]['Promo2SinceWeek']
        self.promoStartYear = storeData[storeData['Store']==storeIndx]['Promo2SinceYear']
        self.promoInterval = storeData[storeData['Store']==storeIndx]['PromoInterval']

        self.competitionDistance = storeData[storeData['Store']==storeIndx]['CompetitionDistance']
        self.competitionStartMonth = storeData[storeData['Store']==storeIndx]['CompetitionOpenSinceMonth']
        self.competitionStartYear = storeData[storeData['Store']==storeIndx]['CompetitionOpenSinceYear']

        missingTime

        #scipy.signal.lombscargle