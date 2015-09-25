from regularized_log_reg import RegularizedLogReg
import numpy as np

class AdvancedLogReg(RegularizedLogReg):
	def test_something(self):
		print self.m, self.n
		print self.X
		print self.y

	'''def oneVsAll(self, theta_len, lambda_val):
		# Optimizes theta for a non-binary set of discrete values.  Doesn't work right now.
		for i in range(theta_len):
			self.y = self.y.isin([i])
			optimal_theta = fmin_cg(self.costFunctionReg, x0=pd.Series(np.zeros(self.n)), args=(lambda_val,))
			print optimal_theta'''
