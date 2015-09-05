#!/usr/bin/python

from multi_lin_reg import MultiLinReg
from numpy import dot, zeros
from pandas import Series

# Example using csv input
df1 = MultiLinReg('csv', col_labels = ['sqr_ft', 'bdrm_count', 'price'], input_file='multi_lin_reg_data.csv')
df1_nrml = df1.featureNormalize()
df1.computeCostMulti(Series(zeros(3)))
df1.gradientDescentMulti(df1_nrml, 1, 50)
theta = df1.normalEquation()

print 'Cost of a 3 bdrm 1650 sqr ft house = ' + str(dot(Series([1, 1650, 3]), theta))

# Example using mySQL input
df2 = MultiLinReg('mySQL', sql_query='SELECT sqr_ft, br_count, price FROM multi_lin_reg_data')
df2_nrml = df1.featureNormalize()
df2.computeCostMulti(Series(zeros(3)))
df2.gradientDescentMulti(df2_nrml, 1, 50)
theta = df1.normalEquation()

print 'Cost of a 3 bdrm 1650 sqr ft house = ' + str(dot(Series([1, 1650, 3]), theta))
