#!/usr/bin/python

from regularized_log_reg import RegularizedLogReg
from numpy import zeros
from pandas import Series

# example using csv input
df1 = RegularizedLogReg(source='csv', col_labels=['microchip_test_1', 'microchip_test_2', 'y'], input_file='log_reg_data2.csv')
# Visualize data
df1.plotData(0)
# Re-create dataframe with polynomial feature because a linear function is not suitable for this dataset
df1 = RegularizedLogReg('csv', col_labels=['microchip_test_1', 'microchip_test_2', 'y'], input_file='log_reg_data2.csv', map_feature=True, degree=6)
df1.costFunctionReg(Series(zeros(28)))
theta = df1.getOptimalTheta(28, 0)
df1.predict(theta)

# example using mySQL input
df2 = RegularizedLogReg(source='mySQL', sql_query='SELECT microchip_test_1, microchip_test_2, y FROM log_reg_data2')
# Visualize data
df2.plotData(0)
# Re-create dataframe with polynomial feature because a linear function is not suitable for this dataset
df2 = RegularizedLogReg(source='mySQL', sql_query='SELECT microchip_test_1, microchip_test_2, y FROM log_reg_data2', map_feature=True, degree=6)
df2.costFunctionReg(Series(zeros(28)))
theta = df2.getOptimalTheta(28, 0)
df2.predict(theta)