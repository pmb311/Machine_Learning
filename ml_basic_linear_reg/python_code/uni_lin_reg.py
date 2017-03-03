from __future__ import division
from my_sql_conn import mySQLconn
from pyspark import SparkContext
from pyspark.sql import *
import pyspark.sql.functions as psf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_data_from_source import GetDataFromSource
# sc = SparkContext('local', 'pyspark')

class UniLinReg(GetDataFromSource):
    def plotDataExample(self, presentation=False):
        if presentation:
            fig = plt.figure()
        plt.scatter(self.X['Population'], self.y)
        plt.xlabel('Population')
        plt.ylabel('Profit')
        plt.show()
        if presentation:
            fig.suptitle('Profit By Population')
            fig.savefig('test.jpg')

    def computeCost(self, theta):
        m = len(self.y)
        h = pd.Series(np.dot(self.X, theta))
        sqrErrors = (h - self.y) ** 2
        J = 1 / (2 * m) * sum(sqrErrors)
        return J

    def sparkComputeCost(self, input_file, x, y, theta):
        df = pd.read_csv(input_file, names=[x, y])
        m = len(df[y])
        df.insert(0,'Ones', pd.Series(np.ones(m)))
        df.insert(0,'h', pd.Series(np.dot(df.ix[:,'Ones':x], theta)))
        sqlCtx = SQLContext(sc)
        spark_df = sqlCtx.createDataFrame(df)
        spark_df = spark_df.withColumn('sqrErrors', psf.pow(spark_df.h - spark_df.Profit, 2)) #FIXME hardcoded to test example
        J = spark_df.select(1 / (2 * m) * psf.sum(spark_df.sqrErrors))
        J.show()
        
    def gradientDescent(self, alpha, num_iters, debug=False):
        X = self.X
        y = self.y
        
        m = len(y)
        J_history = pd.Series(np.zeros(len(y)))
        theta = pd.Series(np.zeros(len(X.columns)))
        for i in range(num_iters):
            theta = theta - (alpha / m) * np.dot(X.T, pd.Series(np.dot(X, theta)) - y)
            J_history[i] = self.computeCost(theta)
        if debug:
            print "Theta\n"
            print theta
            print "J_history\n"
            print J_history
        # graph it
        plt.scatter(X['Population'], y)
        plt.plot(X, theta[1] * X + theta[0], '-')
        plt.xlabel('Population')
        plt.ylabel('Profit')
        plt.show()
        predict1 = 10000 * sum(pd.Series([1, 3.5]) * theta)
        predict2 = 10000 * sum(pd.Series([1, 7]) * theta)
        print 'For population = 35,000, we predict a profit of ' + str(predict1)
        print 'For population = 70,000, we predict a profit of ' + str(predict2)

if __name__ == "__main__":
    print "Running test case..."
    uniLinReg = UniLinReg('mySQL', sql_query='select Population, Profit from uni_lin_reg_data')
    uniLinReg.plotDataExample(True)
    uniLinReg.computeCost(pd.Series(np.zeros(2))) 
    uniLinReg.gradientDescent(0.01, 1500, debug=True)
    print "Done test case..."

# sparkComputeCost('uni_lin_reg_data.csv', 'Population', 'Profit', pd.Series(np.zeros(2)))