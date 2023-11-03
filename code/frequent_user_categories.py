import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.fpm import FPGrowth

# Define new CSVs
good_reviews_csv_path = os.getcwd() + "/../data/csv/good_reviews.csv"
user_preferred_businesses_csv_path = os.getcwd() + "/../data/csv/user_preferred_businesses.csv"
user_preferred_business_categories_csv_path = os.getcwd() + "/../data/csv/user_preferred_business_categories.csv"

# Get business categories
business_df = pd.read_csv(os.getcwd() + "/../data/csv/preprocessed_business.csv")
business_categories_df = business_df[['business_id', 'categories']]

# Remove business with empty categories
business_categories_df = business_categories_df.dropna(subset = ['categories'])

# Read & filter review data
data_reader = pd.read_csv(os.getcwd() + "/../data/csv/review.csv", chunksize=10**4)
for chunk in data_reader:

    # Extract all 4+ stars reviews from chunk
    good_reviews = chunk[chunk['stars']>4.0]

    # Isolate only user_id and business_id
    user_preferred_businesses = good_reviews[['user_id', 'business_id']]
    # print(good_reviews)

    # Get categories from preferred businesses
    user_preferred_business_categories = user_preferred_businesses.merge(business_categories_df, on='business_id', how='left')
    # print(user_preferred_business_categories)

    # Add extracted feature to CSVs
    good_reviews.to_csv(good_reviews_csv_path, mode='a', header=not os.path.exists(good_reviews_csv_path))
    user_preferred_businesses.to_csv(user_preferred_businesses_csv_path, mode='a',
                                           header=not os.path.exists(user_preferred_businesses_csv_path))
    user_preferred_business_categories.to_csv(user_preferred_business_categories_csv_path, mode='a',
                                           header=not os.path.exists(user_preferred_business_categories_csv_path))

# Get complete user categories (by business)
user_preferred_business_categories = pd.read_csv(user_preferred_business_categories_csv_path)

# Group by user_id
preferred_categories_by_user = user_preferred_business_categories.groupby(['user_id'], as_index=False).agg({'categories': ', '.join})
print(preferred_categories_by_user)

# Isolate categories
user_preferred_categories = preferred_categories_by_user['categories']

