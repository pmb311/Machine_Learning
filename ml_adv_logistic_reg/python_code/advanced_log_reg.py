from get_data_from_source import GetDataFromSource
from regularized_log_reg import RegularizedLogReg
import matplotlib.pyplot as plt

class AdvancedLogReg(RegularizedLogReg):
	def test_something(self):
		print self.m, self.n
		print self.X

	# def oneVsAll(self, theta_len, lambda_val):