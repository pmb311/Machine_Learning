from __future__ import division
from numpy import mean, std
import numpy as np
import pandas as pd

def featureNormalize(col_labels, source, sql_query=None, input_file=None):
    if source == 'csv':
        df = pd.read_csv(input_file, names=col_labels).drop(col_labels[-1], 1)
	mu = pd.Series(np.zeros(2))
	sigma = pd.Series(np.zeros(2))
	# calculate z-score for all columns
	cols = list(df.columns)
	df_norm = pd.DataFrame()
	for col in cols:
		col_zscore = col + '_normalized'
		df_norm[col_zscore] = (df[col] - df[col].mean()) / df[col].std()
	print df_norm


print pd.read_csv('multi_lin_reg_data.csv', header=None, names=['sqr ft', 'BRs', 'price']).head(10)

featureNormalize(['sqr ft', 'BRs', 'price'], 'csv', input_file='multi_lin_reg_data.csv')