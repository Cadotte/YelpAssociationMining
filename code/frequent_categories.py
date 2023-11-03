import os
import pandas as pd
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

import numpy as np

spark = SparkSession \
.builder \
.appName("Python Spark SQL basic example") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()


def businessMining(spark:SparkSession,  minsup=0.1, minconf=0.1):

    # Isolate restaurant categories
    preprocessed_business_df = pd.read_csv(os.getcwd() + "/../data/csv/preprocessed_business.csv")
    restaurant_categories_df = preprocessed_business_df['categories']
    restaurant_categories_df.to_csv(os.getcwd() + "/../data/csv/restaurant_categories.csv", index=False, header=False)

    # Read data (restaurant categories)
    data = (spark.read.csv(os.getcwd() + "/../data/csv/restaurant_categories.csv").select(split("_c0", ", ").alias("items")))
    data.show()

    # Find frequent categories (itemsets)
    s = np.arange(0.05, 0.3, 0.05)
    f = np.arange(0.1, 0.5, 0.1)
    itmp = 0
    for i in s:
        for y in f:

            fp = FPGrowth(minSupport=i, minConfidence=y)
            fpm = fp.fit(data)
            fpm.setPredictionCol("newPrediction")
            if(itmp != i):
                fpm.freqItemsets.toPandas().to_csv(os.getcwd() + "/../data/csv/business_best_categories_sup_" + str(i) +".csv")
                itmp = i
            fpm.associationRules.toPandas().to_csv(os.getcwd() + "/../data/csv/business_best_rule_sup_" + str(i) + "_conf_"+ str(y) +".csv")

def mtlBusinessMining(spark:SparkSession):

    # Read data (restaurant categories)
    data = (spark.read.csv(os.getcwd() + "/../data/csv/mtl_business_categories.csv").select(split("_c0", ", ").alias("items")))

    # Find frequent categories (itemsets)
    s = np.arange(0.01, 0.11, 0.01)
    f = np.arange(0.1, 0.5, 0.1)
    itmp = 0
    for i in s:
        for y in f:

            fp = FPGrowth(minSupport=i, minConfidence=y)
            fpm = fp.fit(data)
            fpm.setPredictionCol("newPrediction")
            if(itmp != i):
                fpm.freqItemsets.toPandas().to_csv(os.getcwd() + "/../data/csv/mtl_best_categories_sup_" + str(i) +".csv")
                itmp = i
            fpm.associationRules.toPandas().to_csv(os.getcwd() + "/../data/csv/mtl_best_rule_sup_" + str(i) + "_conf_"+ str(y) +".csv")

def topCityBusinessMining(spark:SparkSession):

    # Read data (restaurant categories)
    data = (spark.read.csv(os.getcwd() + "/../data/csv/top_city_top_business_categories.csv").select(split("_c0", ", ").alias("items")))

    # Find frequent categories (itemsets)
    s = np.arange(0.01, 0.11, 0.01)
    f = np.arange(0.1, 0.5, 0.1)
    itmp = 0
    for i in s:
        for y in f:

            fp = FPGrowth(minSupport=i, minConfidence=y)
            fpm = fp.fit(data)
            fpm.setPredictionCol("newPrediction")
            if(itmp != i):
                fpm.freqItemsets.toPandas().to_csv(os.getcwd() + "/../data/csv/top_city_top_business_best_categories_sup_" + str(i) +".csv")
                itmp = i
            fpm.associationRules.toPandas().to_csv(os.getcwd() + "/../data/csv/top_city_top_business_best_rule_sup_" + str(i) + "_conf_"+ str(y) +".csv")



def userMining(spark:SparkSession):
    #Isolate user categories
    df = spark.read.csv(os.getcwd() + "/../data/csv/user_categories.csv", header=True)
    df = df.select(df.categories.alias("items")).withColumn("items", F.regexp_replace("items", "Restaurants[,]?", ""))\
        .withColumn("items", F.regexp_replace("items", '[{\[\]}]', "")).select(split("items", ", ").alias("items"))

    df = df.withColumn("items", F.array_distinct("items"))

    #computing most popular categories
    s = [0.3,0.2,0.15,0.1]
    f = np.arange(0.1, 0.5, 0.1)
    itmp = 0
    for i in s:
        for y in f:

            fp = FPGrowth(minSupport=i, minConfidence=y)
            fpm = fp.fit(df)
            fpm.setPredictionCol("newPrediction")
            if(itmp != i):
                fpm.freqItemsets.toPandas().to_csv(os.getcwd() + "/../data/csv/user_best_categories_sup_" + str(i) +".csv")
                itmp = i
            fpm.associationRules.toPandas().to_csv(os.getcwd() + "/../data/csv/user_best_rule_sup_" + str(i) + "_conf_"+ str(y) +".csv")


def preciseUserMining(spark:SparkSession):

    df = spark.read.csv(os.getcwd() + "/../data/csv/user2_categories.csv", header=True)
    df = df.withColumn("categories", F.regexp_replace("categories", "\{", ";\{")).withColumn("categories", F.regexp_replace("categories", "[{\[\]}]", ""))\
    .withColumnRenamed("categories", "items")
    df.printSchema()
    df = df.select(split(F.col("items"), ";").alias("items"))
    df.printSchema()
    df = df.withColumn("items", F.array_distinct("items")).filter(F.size("items")> 3)

    s = np.arange(0.01, 0.1, 0.01)
    f = np.arange(0.1, 0.5, 0.1)
    itmp = 0
    for i in s:
        for y in f:

            fp = FPGrowth(minSupport=i, minConfidence=y)
            fpm = fp.fit(df)
            fpm.setPredictionCol("newPrediction")
            if(itmp != i):
                fpm.freqItemsets.toPandas().to_csv(os.getcwd() + "/../data/csv/user_best_reviews_categories_sup_" + str(i) +".csv")
                itmp = i
            fpm.associationRules.toPandas().to_csv(os.getcwd() + "/../data/csv/user_best_reviews_rule_sup_" + str(i) + "_conf_"+ str(y) +".csv")




if __name__ == "__main__":
    if(len(sys.argv) >= 2 and len(sys.argv) <= 4):
        if(sys.argv[1] =="user"):
            userMining(spark)
        elif(sys.argv[1] =="business"):
            businessMining(spark)
        elif(sys.argv[1] =="user2"):
            preciseUserMining(spark)
        elif(sys.argv[1] =="mtl"):
            mtlBusinessMining(spark)
        elif(sys.argv[1] =="cities"):
            mtlBusinessMining(spark)
            topCityBusinessMining(spark)
    else : 
        raise ValueError("No the correct number of args here")