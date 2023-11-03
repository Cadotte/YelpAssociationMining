import os
import sys
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, Row

def build_users_profile_from_reviews_dataset():
    """Builds the dataset for the users from the comments."""

    spark = SparkSession \
    .builder \
    .appName("User database from reviews") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

    #read the reviews
    df_user = spark.read.json(os.getcwd()  + "/../data/json/yelp_academic_dataset_review.json")
    #filter the reviews
    df_user2 = df_user.filter(df_user.stars >= 3).select(["user_id", "business_id"])
    df_user = df_user.select(["user_id", "business_id"])
    
    #read the business and only select the id and the categories
    df_business = spark.read.json(os.getcwd()  + "/../data/json/yelp_academic_dataset_business.json")
    df_business = df_business.select(df_business.business_id, df_business.categories)
   
    #join the two df and collecte the business categories
    df_user = df_user.join(df_business, df_user.business_id == df_business.business_id).select(df_user.user_id, df_business.categories)
    df_user2 = df_user2.join(df_business, df_user.business_id == df_business.business_id).select(df_user.user_id, df_business.categories)

    #groupe all the categories
    df = df_user.groupBy(df_user.user_id).agg(F.collect_list(F.struct(["categories"])).alias("categories"))
    df2 = df_user2.groupBy(df_user.user_id).agg(F.collect_list(F.struct(["categories"])).alias("categories"))

    #save the df as csv
    df = df.withColumn("categories", F.col("categories").cast('string'))
    df.toPandas().to_csv(os.getcwd() + "/../data/csv/user_categories.csv")

    df2 = df2.withColumn("categories", F.col("categories").cast('string'))
    df2.toPandas().to_csv(os.getcwd() + "/../data/csv/user2_categories.csv")

def build_main_dataset():
    """Builds restaurant categories csv """

    # Remove duplicate categories in each business
    preprocessed_business_df = pd.read_csv(os.getcwd() + "/../data/csv/business.csv")
    preprocessed_business_df['categories'] = preprocessed_business_df.categories.str.split(", ").explode().groupby(level=0).unique().str.join(', ')
    # Save preprocessed data to csv
    preprocessed_business_df.to_csv(os.getcwd() + "/../data/csv/preprocessed_business.csv", index=False)

    # Isolate business categories
    preprocessed_business_df = pd.read_csv(os.getcwd() + "/../data/csv/preprocessed_business.csv")
    business_categories_df = preprocessed_business_df['categories']
    # Remove empty lines
    business_categories_df.replace('', np.nan, inplace=True)
    business_categories_df.dropna(inplace=True)
    business_categories_df.reset_index(drop=True, inplace=True)
    # Save business categories to csv
    business_categories_df.to_csv(os.getcwd() + "/../data/csv/business_categories.csv", index=False, header=False)

def build_secondary_dataset():
    """ Builds the data set (in json format) of the city that appears the most"""

    #builds spark session
    spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

    # Read data 
    df = spark.read.json(os.getcwd() + "/../data/json/yelp_academic_dataset_business.json")

    #creating the category file for the top city's business
    top_city = df.groupBy("city").count().orderBy("count", ascending=False).first()
    df2 = df.filter(df.city == top_city[0])
    df2.toPandas().to_csv(os.getcwd() + "/../data/csv/top_city_business.csv")
    df2.filter(df.stars >= 4.0).toPandas().to_csv(os.getcwd() + "/../data/csv/top_city_top_business.csv")

    #creating the category file for mtl's top business
    df2 = df.filter(df.city == "Montréal")
    df2.toPandas().to_csv(os.getcwd() + "/../data/csv/mtl_business.csv")
    df2.filter(df.stars >= 4.0).toPandas().to_csv(os.getcwd() + "/../data/csv/top_mtl_business.csv")

def splitCategories():

    for filename in os.listdir(os.getcwd() + "/../data/csv/"):
        if "_business" in filename:
            if not ("categories" in filename or "preprocessed" in filename) :
                print (filename)
                # Isolate restaurant categories
                preprocessed_business_df = pd.read_csv(os.getcwd() + "/../data/csv/" + filename)
                preprocessed_business_df.head(2)
                preprocessed_business_df['categories'] = preprocessed_business_df.categories.str.split(", ").explode().groupby(level=0).unique().str.join(', ')
                restaurant_categories_df = preprocessed_business_df['categories']
                restaurant_categories_df.to_csv(os.getcwd() + "/../data/csv/" + filename.replace(".csv", "_categories.csv"), index=False, header=False)



if __name__ == "__main__":

    if(len(sys.argv) == 2):
        if(sys.argv[1] =="all"):
            build_main_dataset()
            build_secondary_dataset()
            splitCategories()
            build_users_profile_from_reviews_dataset()
        elif(sys.argv[1] =="main"):
            build_main_dataset()
        elif(sys.argv[1] =="secondary"):
            build_secondary_dataset()
            splitCategories()
        elif(sys.argv[1] =="split"):
            splitCategories()
        elif(sys.argv[1] =="reviews"):
            build_users_profile_from_reviews_dataset()
        else:
            raise ValueError("This argument does not exist!")
    


