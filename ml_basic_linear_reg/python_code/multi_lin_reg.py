from __future__ import division
from get_data_from_source import GetDataFromSource
from numpy import dot, mean, std
import numpy as np
import pandas as pd

class MultiLinReg(GetDataFromSource):
	def featureNormalize(self):
		'''Standard feature normalization algorithm'''
		df = self.X
		# drop the ones column
		df = df.drop('Ones', 1)
		# initialize variables, calculate z-score for all feature columns, and append them to a new dataframe
		cols = list(df.columns)
		df_norm = pd.DataFrame()
		for col in cols:
			col_zscore = col + '_normalized'
			df_norm[col_zscore] = (df[col] - df[col].mean()) / df[col].std()
		# add the ones column onto the normalized dataframe
		m = len(df_norm.ix[:, 0])
		df_norm.insert(0,'Ones', pd.Series(np.ones(m)))
		print 'Printing normalized features:\n'
		print df_norm
		return df_norm

	def computeCostMulti(self, theta):
		'''Compute the cost for input data containing any number of features'''
		# initialize variables
		x = self.X
		y = self.y
		m = self.m
		# calculate the cost and assign the result to J
		h = pd.Series(dot(x, theta))
		sqrErrors = (h - y) ** 2
		J = 1 / (2 * m) * sum(sqrErrors)
		print 'Cost = ' + str(J)
		return J

	def gradientDescentMulti(self, normalized_df, alpha, num_iters):
		'''Gradient descent for input data containing any number of features'''
		X = normalized_df
		y = self.y
		m = self.m
		# we will keep track of the costs in J_history
		J_history = pd.Series(np.zeros(len(X)))
		# Add ones column and initialize theta
		theta = pd.Series(np.zeros(len(X.columns)))
		# Do gradient descent
		for i in range(num_iters):
			theta = theta - (alpha / m) * dot(X.T, pd.Series(dot(X, theta)) - y)
			# FIXME insert cost into J_History J_history[i] = self.df.computeCostMulti(theta)
		print 'Final theta values:\n' + str(list(theta))
		return list(theta)

	def normalEquation(self):
		'''Standard normal equation.  Preferable to gradient descent for datasets containing less than 1000-10000 features'''
		# initialize variables
		m = self.m
		x = self.X
		y = self.y
		# perform normal equation
		theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
		print "Theta found from normal equation:\n" + str(theta)
		return theta
