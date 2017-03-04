from __future__ import division
from my_sql_conn import mySQLconn
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix, BlockMatrix
from pyspark.sql import *
import pyspark.sql.functions as psf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_data_from_source import GetDataFromSource

class UniLinReg(GetDataFromSource):
    def computeCost(self, theta):
        m = len(self.y)
        h = pd.Series(np.dot(self.X, theta))
        sqrErrors = (h - self.y) ** 2
        J = 1 / (2 * m) * sum(sqrErrors)
        return J

    def sparkComputeCost(self, input_file, x, y, theta):
        
        sc = SparkContext()

        # add the ones vector while building the RDD
        idx = 0
        x_mat = sc.textFile(input_file) \
            .map(lambda line: ('1, ' + line).split(",")[:-1]) \
            .zipWithIndex()
        
        # need a SQLContext() to generate an IndexedRowMatrix from RDD
        sqlContext = SQLContext(sc)
        
        x_mat = IndexedRowMatrix( \
            x_mat \
            .map(lambda row: IndexedRow(row[1], row[0])) \
            ).toBlockMatrix()

        x_mat.cache()

        print "Matrix rows x cols"
        print x_mat.numRows()
        print x_mat.numCols()

        vec = sc.parallelize(theta) \
            .map(lambda line: [line]) \
            .zipWithIndex()

        vec = IndexedRowMatrix( \
            vec \
            .map(lambda row: IndexedRow(row[1], row[0])) \
            ).toBlockMatrix()

        vec.cache()

        print "Vector rows x cols"
        print vec.numRows()
        print vec.numCols()

        h = x_mat.multiply(vec)
        h.cache()

        print "Hypothesis rows x cols"
        print h.numRows()
        print h.numCols()

        y_vec = sc.textFile(input_file) \
            .map(lambda line: [('1, ' + line).split(",")[-1]]) \
            .zipWithIndex()

        y_vec = IndexedRowMatrix( \
            y_vec \
            .map(lambda row: IndexedRow(row[1], row[0])) \
            ).toBlockMatrix()

        y_vec.cache()

        errors = h.subtract(y_vec).toLocalMatrix()

        print sum(errors.toArray())

        '''sparkSession = SparkSession \
            .builder \
            .appName('pyspark') \
            .getOrCreate()
        
        df = sparkSession.read.csv(input_file)
        df = df \
            .toDF(x, y) \
            .withColumn("Ones", psf.lit(1)) \
            .cache()

        df.select(x,'Ones').show()'''

        '''sc = SparkContext('local', 'pyspark')
        df = pd.read_csv(input_file, names=[x, y])
        m = len(df[y])
        df.insert(0,'Ones', pd.Series(np.ones(m)))
        df.insert(0,'h', pd.Series(np.dot(df.ix[:,'Ones':x], theta)))
        sqlCtx = SQLContext(sc)
        spark_df = sqlCtx.createDataFrame(df)
        spark_df = spark_df.withColumn('sqrErrors', psf.pow(spark_df.h - spark_df.Profit, 2)) #FIXME hardcoded to test example
        J = spark_df.select(1 / (2 * m) * psf.sum(spark_df.sqrErrors))
        J.show()'''
        
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

if __name__ == "__main__":
    print "Running test case..."
    uniLinReg = UniLinReg('mySQL', sql_query='select Population, Profit from uni_lin_reg_data')
    # uniLinReg.plotDataExample(True)
    print uniLinReg.computeCost(pd.Series(np.zeros(2))) 
    uniLinReg.gradientDescent(0.01, 1500, debug=True)
    print "Done test case..."

    uniLinReg.sparkComputeCost('uni_lin_reg_data.csv', 'Population', 'Profit', pd.Series(np.zeros(2)))