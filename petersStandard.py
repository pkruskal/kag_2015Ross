import pandas as pd
import numpy as np




#useful definition for looking at a data frame
def exploreDF(data):
    '''
    Display a data frames data
    '''
    from PyQt4 import QtGui
    datatable = QtGui.QTableWidget()
    datatable.setColumnCount(len(data.columns))
    datatable.setRowCount(len(data.index))
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            datatable.setItem(i,j,QtGui.QTableWidgetItem(str(data.iget_value(i, j))))
    datatable.show()
    return datatable