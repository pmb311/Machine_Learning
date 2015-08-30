from __future__ import division
from my_sql_conn import mySQLconn
from numpy import dot
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import pylab as pl

class RegularizedLogReg():

	df = pd.DataFrame

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
				RegularizedLogReg.df = pd.read_csv(input_file, names=col_labels)
			except IOError:
				print 'Error: input_file is required for this source'
		elif source == 'mySQL':
			try:
				conn = mySQLconn
				RegularizedLogReg.df = pd.read_sql(sql_query, conn)
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

	def plotData(self, theta):
		'''Hard-coded for sample data'''
		df = self.df
		df_pos = df.loc[df['y'] == 1]
		df_neg = df.loc[df['y'] == 0]
		pl.scatter(df_pos['Microchip Test 1'], df_pos['Microchip Test 2'], marker='+', c='b')
		pl.scatter(df_neg['Microchip Test 1'], df_neg['Microchip Test 2'], marker='o', c='r')
		pl.xlabel('Microchip Test 1')
		pl.ylabel('Microchip Test 2')
		pl.legend(['y = 1', 'y = 0'])
		pl.show()

df1 = RegularizedLogReg('csv', col_labels=['Microchip Test 1', 'Microchip Test 2', 'y'], input_file='log_reg_data2.csv')
df1.plotData(0)	