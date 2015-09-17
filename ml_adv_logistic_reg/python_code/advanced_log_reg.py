from get_data_from_source import GetDataFromSource
from regularized_log_reg import RegularizedLogReg
from scipy.optimize import fmin_cg
import numpy as np
import pandas as pd

class AdvancedLogReg(RegularizedLogReg):
	def test_something(self):
		print self.m, self.n
		print self.X
		print self.y

	'''def oneVsAll(self, theta_len, lambda_val):
		# Optimizes theta for a non-binary set of discrete values.
		for i in range(theta_len):
			self.y = self.y.isin([i])
			optimal_theta = fmin_cg(self.costFunctionReg, x0=pd.Series(np.zeros(theta_len)), args=(lambda_val,))
			print optimal_theta'''