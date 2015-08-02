from __future__ import division
from my_sql_conn import mySQLconn
from pyspark import SparkContext
from pyspark.sql import *
import pyspark.sql.functions as psf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sc = SparkContext('local', 'pyspark')

def warmUpExercise():
    print np.identity(5)

def plotData(input_file, x, y, presentation=False):
    df = pd.read_csv(input_file, names=[x, y])
    if presentation:
        fig = plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.show()
    if presentation:
        fig.suptitle('Profit By Population')
        fig.savefig('test.jpg')

def computeCost(input_file, x, y, theta):
    df = pd.read_csv(input_file, names=[x, y])
    m = len(df[y])
    df.insert(0,'Ones', pd.Series(np.ones(m)))
    h = pd.Series(np.dot(df.ix[:,'Ones':x], theta))
    sqrErrors = (h - df[y]) ** 2
    J = 1 / (2 * m) * sum(sqrErrors)
    print J

def sparkComputeCost(input_file, x, y, theta):
    df = pd.read_csv(input_file, names=[x, y])
    m = len(df[y])
    df.insert(0,'Ones', pd.Series(np.ones(m)))
    df.insert(0,'h', pd.Series(np.dot(df.ix[:,'Ones':x], theta)))
    sqlCtx = SQLContext(sc)
    spark_df = sqlCtx.createDataFrame(df)
    spark_df = spark_df.withColumn('sqrErrors', psf.pow(spark_df.h - spark_df.Profit, 2)) #FIXME hardcoded to test example
    J = spark_df.select(1 / (2 * m) * psf.sum(spark_df.sqrErrors))
    J.show()
    

def gradientDescent(input_file, x, y, alpha, num_iters, source, sql_query=None):
    if source == 'csv':
        df = pd.read_csv(input_file, names=[x, y])
    if source == 'mySQL':
        conn = mySQLconn
        df = pd.read_sql(sql_query, conn)
    m = len(df[y])
    df.insert(0,'Ones', pd.Series(np.ones(m)))
    J_history = pd.Series(np.zeros(len(df)))
    X = df.ix[:,'Ones':x].T
    theta = pd.Series(np.zeros(len(df.columns) - 1))
    for i in range(num_iters):
        theta = theta - (alpha / m) * np.dot(X, pd.Series(np.dot(df.ix[:,'Ones':x], theta)) - df.ix[:, y])
    print theta

# warmUpExercise()

# plotData('uni_lin_reg_data.csv', 'Population', 'Profit')

computeCost('uni_lin_reg_data.csv', 'Population', 'Profit', pd.Series(np.zeros(2)))

# gradientDescent('uni_lin_reg_data.csv', 'Population', 'Profit', 0.01, 1500, 'mySQL', 'select Population, Profit from uni_lin_reg_data')

sparkComputeCost('uni_lin_reg_data.csv', 'Population', 'Profit', pd.Series(np.zeros(2)))