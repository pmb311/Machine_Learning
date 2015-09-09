from __future__ import division
from get_data_from_source import GetDataFromSource
from numpy import dot
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import pylab as pl

class RegularizedLogReg(GetDataFromSource):
	def __init__(self, source, col_labels=None, sql_query=None, input_file=None, map_feature=False, degree=0):
		super(RegularizedLogReg, self).__init__(source, col_labels, sql_query, input_file)
		# Do mapFeature if specified
		if map_feature:
			self.X = self.mapFeature(degree)

	def mapFeature(self, degree):
		'''Adds polynomial features to the training set.  Currently only supports 2-feature input.'''
		X1 = self.X.ix[:, 1]
		X2 = self.X.ix[:, 2]
		length = len(X1.index)
		# Start with dataframe of just a 1's column.  We will append calculated features to it.
		out = pd.DataFrame(np.ones(length), columns=['calc_feature_1'])
		count = 2
		for i in range(1, degree + 1):
			for j in range(0, i + 1):
				out.insert(len(out.axes[1]), 'calc_feature_' + str(count), (X1 ** (i - j)) * (X2 ** j))
				count += 1
		print 'New dataframe:\n' + str(out)
		return out

	def costFunctionReg(self, theta, lambda_val=0, X=None, y=None):
		'''Regularized cost function.'''
		# Initialize variables for computing cost function
		X = self.X
		m, n = X.shape
		y = self.y
		# Create hypothesis with sigmoid function
		h = 1 / (1 + np.exp(dot(X, theta)))
		# calculate non-regularized cost
		cost = sum(((-y * np.log(h)) - ((1 - y) * np.log(1 - h)))) / m
		# Set up control row for regularization.  All ones except the first element.
		control = pd.Series(np.ones(len(theta))).set_value(0, 0)
		# Calculate regularization parameter for cost function
		reg = (lambda_val / (2 * m)) * sum((theta * control) ** 2)
		J = cost + reg 
		# Calculate regularization parameter for gradient
		reg_g = (lambda_val / m) * (theta * control)
		# Run regularized gradient descent
		for i in range(n):
			grad = ((1 / m) * X.T.dot(h - y)) + reg_g.values
		print 'Cost = ' + str(J)
		print 'Gradient:\n' + str(list(grad))
		return J

	def getOptimalTheta(self, theta_len, lambda_val):
		'''Use the scipy optimize method minimize to obtain optimal theta'''
		optimal_theta = minimize(self.costFunctionReg, x0=pd.Series(np.zeros(theta_len)), args=(lambda_val,), method='TNC', jac=False)
		print 'Optimal theta:\n' + str(optimal_theta.x)
		return optimal_theta.x

	def predict(self, theta):
		'''Compare accuracy of theta vs. training set'''
		X = self.X
		y = self.y
		p = 100 * np.mean((np.round(pd.Series(1 / (1 + np.exp(dot(X, theta))))) == y).convert_objects(convert_numeric=True))
		print 'Training accuracy = ' + str(p)
		return p

	def plotData(self, theta):
		''' FIXME Hard-coded for sample data'''
		df = self.df
		df_pos = df.loc[df['y'] == 1]
		df_neg = df.loc[df['y'] == 0]
		pl.scatter(df_pos['microchip_test_1'], df_pos['microchip_test_2'], marker='+', c='b')
		pl.scatter(df_neg['microchip_test_1'], df_neg['microchip_test_2'], marker='o', c='r')
		pl.xlabel('Microchip Test 1')
		pl.ylabel('Microchip Test 2')
		pl.legend(['y = 1', 'y = 0'])
		pl.show()
