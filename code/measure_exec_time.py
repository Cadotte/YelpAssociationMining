import os
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.fpm import FPGrowth

# mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori

# pyECLAT
from pyECLAT import ECLAT

def spark_fp_growth(data, s):
    # Find frequent categories (itemsets) among businesses
    fp = FPGrowth(minSupport=s, minConfidence=0.1)
    fpm = fp.fit(data)
    fpm.setPredictionCol("newPrediction")
    itemsets_df = fpm.freqItemsets.toPandas()
    return itemsets_df

def sparse_encode(data):
    data = data['items'].str.split(',')
    te = TransactionEncoder()
    fitted = te.fit(data)
    te_ary = fitted.transform(data, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    return df

def onehot_encode(data):
    data = data['items'].str.split(',')
    onehot_binarizer = MultiLabelBinarizer()
    onehot_array = onehot_binarizer.fit_transform(data)
    onehot_df = pd.DataFrame(onehot_array, columns=onehot_binarizer.classes_)
    return onehot_df

def mlxtend_fp_frowth(data, s):
    df = sparse_encode(data)
    # Measure exec time
    start_time = time.time()
    itemsets_df = fpgrowth(df, min_support=s, use_colnames=True)
    end_time = time.time()
    exec_time = end_time - start_time
    itemsets_df = itemsets_df.rename(columns={"itemsets": "items"})
    print(itemsets_df)
    return itemsets_df, exec_time

def mlxtend_apriori(data, s):
    df = sparse_encode(data)
    # Measure exec time
    start_time = time.time()
    itemsets_df = apriori(df, min_support=s, use_colnames=True)
    end_time = time.time()
    exec_time = end_time - start_time
    itemsets_df = itemsets_df.rename(columns={"itemsets": "items"})
    print(itemsets_df)
    return itemsets_df, exec_time

def pyeclat_wrapper(data, s):
    data['items'] = data['items'].str.split(',')
    print(data)
    df = pd.DataFrame(data['items'].tolist())
    print(df)
    eclat_instance = ECLAT(data=df, verbose=True)
    # Measure exec time
    start_time = time.time()
    indexes, supports = eclat_instance.fit(min_support=s, min_combination=1, max_combination=1, separator=' & ',
                                     verbose=True)
    end_time = time.time()
    exec_time = end_time - start_time
    itemsets_df = pd.DataFrame(supports.keys(), columns=['items'])
    print(itemsets_df)
    return itemsets_df, exec_time

def measure_sup_range(sup_range, data, algo):
    exec_times = pd.DataFrame(columns=['support', 'exec_time', 'nb_freq_itemsets', 'max_itemset_size', 'mean_itemset_size'])

    for s in sup_range:

        if algo == 'fp_growth':
            # itemsets_df, exec_time = spark_fp_growth(data, s)
            itemsets_df, exec_time = mlxtend_fp_frowth(data, s)
        elif algo == 'apriori':
            itemsets_df, exec_time = mlxtend_apriori(data, s)
        elif algo == 'eclat':
            itemsets_df, exec_time = pyeclat_wrapper(data, s)

        # Get other metrics
        nb_itemsets = itemsets_df['items'].count()
        itemsets_df['itemset_size'] = itemsets_df['items'].apply(lambda x: len(x))
        max_itemset_size = itemsets_df['itemset_size'].max()
        mean_itemset_size = itemsets_df['itemset_size'].mean()

        # print("FP-Growth execution time : " + str(exec_time) + " seconds")
        measurement = {'support': s, 'exec_time': exec_time, 'nb_freq_itemsets':nb_itemsets, 'max_itemset_size':max_itemset_size, 'mean_itemset_size':mean_itemset_size}
        exec_times = exec_times.append(measurement, ignore_index=True)

    return exec_times

exec_measurements_df = pd.DataFrame(columns=['support', 'fp_exec_time', 'ap_exec_time', 'nb_freq_itemsets', 'max_itemset_size', 'mean_itemset_size'])

# Support range
ap_sup_range = np.arange(0.011, 0.1, 0.001)
print(ap_sup_range)
fp_sup_range = np.linspace(0.0001,0.1,100)
print(fp_sup_range)

# Make measurements
fp_sup_exec_times = measure_sup_range(fp_sup_range, data, 'fp_growth').rename(columns={"exec_time": "fp_exec_time"})
print(fp_sup_exec_times)

# exec_measurements_df = exec_measurements_df.append(sup_exec_times, ignore_index=True)
exec_measurements_df['support'] = fp_sup_exec_times['support']
exec_measurements_df['nb_freq_itemsets'] = fp_sup_exec_times['nb_freq_itemsets']
exec_measurements_df['max_itemset_size'] = fp_sup_exec_times['max_itemset_size']
exec_measurements_df['mean_itemset_size'] = fp_sup_exec_times['mean_itemset_size']
exec_measurements_df['fp_exec_time'] = fp_sup_exec_times['fp_exec_time']

print(exec_measurements_df)
exec_measurements_df.to_csv(os.getcwd() + "/../results/exec_time_measurements.csv", index=False)




