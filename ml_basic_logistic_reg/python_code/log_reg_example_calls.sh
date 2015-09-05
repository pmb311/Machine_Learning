#!/usr/bin/python

from log_reg import LogReg
from numpy import dot, exp, zeros
from pandas import Series

# example using csv input
df1 = LogReg('csv', col_labels=['test1_score', 'test2_score', 'admission_status'], input_file='log_reg_data1.csv')
df1.plotData(0)
df1.costFunction(Series(zeros(3)))
theta = df1.getOptimalTheta()
df1.predict(theta)
print 'For a student with scores 45 and 85, probability of admission = ' + str(1 / (1 + exp(dot(Series([1, 45, 85]), theta))))

# example using mySQL input
df2 = LogReg('mySQL', sql_query='select test1_score, test2_score, admission_status from log_reg_data1')
df2.plotData(0)
df2.costFunction(Series(zeros(3)))
theta = df2.getOptimalTheta()
df2.predict(theta)
print 'For a student with scores 45 and 85, probability of admission = ' + str(1 / (1 + exp(dot(Series([1, 45, 85]), theta))))
