import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import datetime as dt
import re

trainSet = pd.read_csv('../data/train.csv',index_col='Date',parse_dates=True)
testSet = pd.read_csv('../data/test.csv',index_col='Date',parse_dates=True)


#print an inital report of the data
print 'training starts at ' + str(trainSet.index.min())
print 'training ends at ' + str(trainSet.index.max())
print 'predicting starts at ' + str(testSet.index.min())
print 'predicting ends at ' + str(testSet.index.max())

#extract stors
print 'there are ' + str(len(set(trainSet['Store']))) + ' unique stores in the data set'

print 'the general structure and stats of the data '
print trainSet.describe()

#some initial boxplots
trainSet.boxplot('Sales','DayOfWeek')
trainSet.boxplot('Sales','Promo')
trainSet.boxplot('Sales','StateHoliday')
trainSet.boxplot('Sales','SchoolHoliday')


#create a data frame for a single store and plot timeseres
def plotStoresTimeSeries(trainSet,storeID,savepath = None):
    #storeID = 1
    thisStore = trainSet[trainSet['Store'] == storeID]

    #set date as the index
    #thisStore = thisStore.set_index('Date')

    plt.figure(figsize=[20,9])
    plt.subplot(3,1,1)
    plt.title('store ' + str(storeID))
    thisStore[thisStore['Open'] == 1]['Customers'].plot()
    plt.ylabel('Customers')
    plt.subplot(3,1,2)
    thisStore[thisStore['Open'] == 1]['Sales'].plot()
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
        plt.close()


#storeNum = 3
for storeNum in set(trainSet['Store']):
    plotStoresTimeSeries(trainSet,storeNum,'../figures/storeTimeseries/store' + str(storeNum) + '.jpg')



idx = pd.date_range(dt.datetime(2013,1,1,00,00,00),dt.datetime(2015,7,31,00,00,00),freq = 'D')
salesMatDay1 = np.zeros([len(set(trainSet['Store'])), len(idx)])
for irow, storeID in enumerate(set(trainSet['Store'])):
    thisStore = trainSet[trainSet['Store'] == storeID]
    theseSales = thisStore[thisStore['DayOfWeek'] == 1].Sales
    theseSales = theseSales.reindex(idx)
    salesMatDay1[irow,:] = theseSales.values


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


