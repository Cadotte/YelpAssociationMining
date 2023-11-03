import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.fpm import FPGrowth

# FP-Growth w/ Spark
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Read data (business categories)
data = (spark.read.csv(os.getcwd() + "/../data/csv/business_categories.csv").select(split("_c0", ", ").alias("items")))
data.show()

# Find frequent categories (itemsets) among businesses
fp = FPGrowth(minSupport=0.01, minConfidence=0.1)
fpm = fp.fit(data)
fpm.setPredictionCol("newPrediction")
fpm.freqItemsets.show(50)