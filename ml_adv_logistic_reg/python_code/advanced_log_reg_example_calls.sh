#!/usr/bin/python

from advanced_log_reg import AdvancedLogReg
from numpy import zeros
from pandas import Series

# example using .mat file as source data
df1 = AdvancedLogReg(source='mat', input_file='log_reg_data3.mat')
df1.test_something()
