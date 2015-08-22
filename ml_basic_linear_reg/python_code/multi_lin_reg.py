from __future__ import division
from my_sql_conn import mySQLconn
from numpy import dot, mean, std
import numpy as np
import pandas as pd

class MultiLinReg():

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
				MultiLinReg.df = pd.read_csv(input_file, names=col_labels)
			except IOError:
				print 'Error: input_file is required for this source'
		elif source == 'mySQL':
			try:
				conn = mySQLconn
				MultiLinReg.df = pd.read_sql(sql_query, conn)
			except TypeError:
				print 'Error: sql_query is required for this source'
		else:
			print 'Only csv and mySQL sources are currently supported'
			raise TypeError

	def featureNormalize(self):
		df = self.df
		# drop the dependent variable
		df = df.drop(df.columns[[-1]], 1)
		# calculate z-score for all feature columns
		cols = list(df.columns)
		df_norm = pd.DataFrame()
		for col in cols:
			col_zscore = col + '_normalized'
			df_norm[col_zscore] = (df[col] - df[col].mean()) / df[col].std()
		print 'Printing normalized features:\n'
		print df_norm
		return df_norm

	def computeCostMulti(self, theta):
		df = self.df
		# instantiate variables
		x = df.drop(df.columns[-1], 1)
		y = df.ix[:,-1]
		m = len(x.ix[:, 0])
		# add ones column
		x.insert(0,'Ones', pd.Series(np.ones(m)))
		# calculate the cost and assign the result to J
		h = pd.Series(dot(x, theta))
		sqrErrors = (h - y) ** 2
		J = 1 / (2 * m) * sum(sqrErrors)
		print 'Cost = ' + str(J)
		return J

	def gradientDescentMulti(self, normalized_df, alpha, num_iters):
		X = normalized_df
		y = self.df.ix[:,-1]
		m = len(X.ix[:,0])
		# we will keep track of the costs in J_history
		J_history = pd.Series(np.zeros(len(X)))
		# Add ones column and initialize theta
		X.insert(0,'Ones', pd.Series(np.ones(m)))
		theta = pd.Series(np.zeros(len(X.columns)))
		# Do gradient descent
		for i in range(num_iters):
			theta = theta - (alpha / m) * dot(X.T, pd.Series(dot(X, theta)) - y)
			# FIXME insert cost into J_History J_history[i] = self.df.computeCostMulti(theta)
		print 'Final theta values:\n' + str(theta)
		return theta

	def normalEquation(self):
		df = self.df
		m = len(df.ix[:,0])
		df.insert(0,'Ones', pd.Series(np.ones(m)))
		x = df.drop(df.columns[-1], 1)
		y = df.ix[:,-1]
		theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
		print "Theta found from normal equation:\n" + str(theta)
		return theta

df1 = MultiLinReg('csv', col_labels = ['abc', 'def', 'ghi'], input_file='multi_lin_reg_data.csv')
df1_nrml = df1.featureNormalize()
df1.computeCostMulti(pd.Series(np.zeros(3)))
df1.gradientDescentMulti(df1_nrml, 1, 50)
df1.normalEquation()