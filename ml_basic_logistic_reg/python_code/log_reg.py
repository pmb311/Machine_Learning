from __future__ import division
from my_sql_conn import mySQLconn
from numpy import dot
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LogReg():

	df = pd.DataFrame()

	def __init__(self, source, col_labels=None, sql_query=None, input_file=None):
		# initialize variables
		self.source = source
		self.col_labels = col_labels
		self.sql_query = sql_query
		self.input_file = input_file	

		# set up dataframe from mySQL or csv source data
		if source == 'csv':
			if col_labels == None:
				print 'col_labels are required for the source'
				raise TypeError
			try:
				LogReg.df = pd.read_csv(input_file, names=col_labels)
			except IOError:
				print 'Error: input_file is required for this source'
		elif source == 'mySQL':
			try:
				conn = mySQLconn
				LogReg.df = pd.read_sql(sql_query, conn)
			except TypeError:
				print 'Error: sql_query is required for this source'
		else:
			print 'Only csv and mySQL sources are currently supported'
			raise TypeError

		df = self.df
		# initialize variables and prepare dataframes
		self.X = df.drop(df.columns[-1], 1)
		self.y = df.ix[:,-1]
		self.m, self.n = self.X.shape
		self.X.insert(0,'Ones', pd.Series(np.ones(self.m)))

	'''	def plotData(self):
		df = self.df
		x = df.drop(df.columns[-1], 1)
		y = df.ix[:,-1]
		plt.scatter(x[self.col_labels[0]], x[self.col_labels[1]])
		plt.show()'''

	def costFunction(self, theta, X=None, y=None):
		X = self.X
		y = self.y
		m = self.m
		n = self.n
		# create hypothesis with sigmoid function
		h = 1 / (1 + np.exp(dot(X, theta)))
		# calculate cost
		J = sum(((-y * np.log(h)) - ((1 - y) * np.log(1 - h)))) / m
		# calculate gradient
		for i in range(n):
			grad = (1 / m) * X.T.dot(h - y)
		print 'Cost = ' + str(J)
		print 'Gradient:\n' + str(list(grad))
		return J
		
	def getOptimalTheta(self):
		optimal_theta = minimize(self.costFunction, x0=pd.Series(np.zeros(3)), method='TNC', jac=False)
		print 'Optimal theta:\n' + str(optimal_theta.x)
		return optimal_theta.x

df1 = LogReg('mySQL', sql_query='select test1_score, test2_score, admission_status from log_reg_data1')
# df1.plotData()
df1.costFunction(pd.Series(np.zeros(3)))
df1.getOptimalTheta()
print 'For a student with scores 45 and 85, probability of admission = ' + str(1 / (1 + np.exp(dot(pd.Series([1, 45, 85]), df1.getOptimalTheta()))))
