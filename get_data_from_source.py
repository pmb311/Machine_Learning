from my_sql_conn import mySQLconn
from scipy.io import loadmat
import numpy as np
import pandas as pd

class GetDataFromSource(object):

	df = pd.DataFrame()

	def __init__(self, source, col_labels=None, sql_query=None, input_file=None):
		# initialize variables
		self.source = source
		self.col_labels = col_labels
		self.sql_query = sql_query
		self.input_file = input_file

		# set up dataframe from mySQL, csv, or mat source data
		if source == 'csv':
			if col_labels == None:
				print 'col_labels are required for the source'
				raise TypeError
			try:
				GetDataFromSource.df = pd.read_csv(input_file, names=col_labels)
			except IOError:
				print 'Error: input_file is required for this source'
		elif source == 'mySQL':
			try:
				conn = mySQLconn
				GetDataFromSource.df = pd.read_sql(sql_query, conn)
			except TypeError:
				print 'Error: sql_query is required for this source'
		elif source == 'mat':
			try:
				GetDataFromSource.df = loadmat(input_file)
			except TypeError:
				print 'Error: input_file is required for this source'
		else:
			print 'Only csv, mySQL, and mat sources are currently supported'
			raise TypeError

		df = self.df
		# Initialize variables and prepare dataframes.  For mat type, it is assumed that X and y are formalized already.
		if source != 'mat':
			self.X = df.drop(df.columns[-1], 1)
			self.y = df.ix[:,-1]
		else:
			self.X = self.df['X']
			self.y = self.df['y']
		self.m, self.n = self.X.shape
		if source != 'mat':
			self.X.insert(0,'Ones', pd.Series(np.ones(self.m)))