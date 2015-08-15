from __future__ import division
from numpy import mean, std
import numpy as np
import pandas as pd

def featureNormalize(col_labels, source, sql_query=None, input_file=None):
    if source == 'csv':
        df = pd.read_csv(input_file, names=col_labels).drop(col_labels[-1], 1)
	mu = pd.Series(np.zeros(2))
	sigma = pd.Series(np.zeros(2))
	# calculate z-score for all feature columns
	cols = list(df.columns)
	df_norm = pd.DataFrame()
	for col in cols:
		col_zscore = col + '_normalized'
		df_norm[col_zscore] = (df[col] - df[col].mean()) / df[col].std()
	return df_norm

def computeCostMulti(col_labels, theta, df=None, input_file=None):
    if input_file != None:
    	df = pd.read_csv(input_file, names=col_labels)
    x = df.drop(col_labels[-1], 1)
    m = len(x.ix[:, 0])
    x.insert(0,'Ones', pd.Series(np.ones(m)))
    h = pd.Series(np.dot(x, theta))
    sqrErrors = (h - df[col_labels[-1]]) ** 2
    J = 1 / (2 * m) * sum(sqrErrors)
    print J

def gradientDescentMulti(input_file, col_labels, alpha, num_iters, source, sql_query=None):
	# Assumes that final column in the input parameter is the dependent variable
    if source == 'csv':
        df = pd.read_csv(input_file, names=col_labels)
    if source == 'mySQL':
        conn = mySQLconn
        df = pd.read_sql(sql_query, conn)
    # Instantiate some variables
    m = len(df.ix[:,0])
    J_history = pd.Series(np.zeros(len(df)))
    # Normalize the feature matrix
    X = featureNormalize(col_labels, source, sql_query, input_file)
    X.insert(0,'Ones', pd.Series(np.ones(m)))
    theta = pd.Series(np.zeros(len(X.columns)))
    # Do gradient descent
    for i in range(num_iters):
    	theta = theta - (alpha / m) * np.dot(X.T, pd.Series(np.dot(X, theta)) - df[col_labels[-1]])
    print theta

print pd.read_csv('multi_lin_reg_data.csv', header=None, names=['sqr ft', 'BRs', 'price']).head(10)

hs_ft_nrml = featureNormalize(['sqrft', 'BRs', 'price'], 'csv', input_file='multi_lin_reg_data.csv')

computeCostMulti(['sqrft', 'BRs', 'price'], pd.Series(np.zeros(3)), input_file='multi_lin_reg_data.csv')

gradientDescentMulti('multi_lin_reg_data.csv', ['sqrft', 'BRs', 'price'], 1, 50, 'csv')