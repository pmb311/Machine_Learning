from __future__ import division
from get_data_from_source import GetDataFromSource
from numpy import dot
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import pylab as pl

class LogReg(GetDataFromSource):
	def costFunction(self, theta, X=None, y=None):
		'''Logistic regression cost function'''
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
		'''Use the scipy optimize method minimize to obtain optimal theta'''
		optimal_theta = minimize(self.costFunction, x0=pd.Series(np.zeros(3)), method='TNC', jac=False)
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
		df_pos = df.loc[df['admission_status'] == 1]
		df_neg = df.loc[df['admission_status'] == 0]
		pl.scatter(df_pos['test1_score'], df_pos['test2_score'], marker='+', c='b')
		pl.scatter(df_neg['test1_score'], df_neg['test2_score'], marker='o', c='r')
		pl.xlabel('Exam 1 score')
		pl.ylabel('Exam 2 score')
		pl.legend(['Admitted', 'Not Admitted'])
		pl.show()
