import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def sparse_encode(data):
    data = data['items'].str.split(',')
    # print(data)
    te = TransactionEncoder()
    fitted = te.fit(data)
    te_ary = fitted.transform(data, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    # print(df)
    return df

def getFreqItemsetsCurves():

    # Load transactions (business categories)
    transactions = pd.read_csv(os.getcwd() + "/../data/csv/business_categories.csv", names=['items'])

    # For each support
    support_range = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    measurements_df = pd.DataFrame(columns=support_range.astype(str))

    for s in support_range:

        # Get freq items
        df = sparse_encode(transactions)
        itemsets_df = fpgrowth(df, min_support=s, use_colnames=True)
        itemsets_df = itemsets_df.rename(columns={"itemsets": "items"})

        # Group freq items by itemset size
        itemsets_df['itemset_size'] = itemsets_df['items'].apply(lambda x: len(x))
        nb_freq_itemsets_by_size = itemsets_df.groupby('itemset_size').agg(nb_itemsets=pd.NamedAgg(column='items', aggfunc='count')) #.reset_index()
        print(nb_freq_itemsets_by_size)

        # Append measurements dataframe
        measurements_df[str(s)] = nb_freq_itemsets_by_size

    print(measurements_df)
    measurements_df.to_csv(os.getcwd() + "/../results/nb_freq_itemsets_by_size.csv")

def getBusinessCategoriesMetrics():

    # Load transactions (business categories)
    transactions = pd.read_csv(os.getcwd() + "/../data/csv/business_categories.csv", names=['items'])

    # Nb of transactions (businesses)
    print("Number of transactions: " + str(len(transactions.index)))

    # Max/Median transaction size (in number of items/categories)
    transactions['items'] = transactions['items'].str.split(',')
    transactions['transaction_size'] = transactions['items'].apply(lambda x: len(x))
    max_transaction_size = transactions['transaction_size'].max()
    mean_transaction_size = transactions['transaction_size'].mean()
    print("Max transaction size: " + str(max_transaction_size))
    print("Mean transaction size: " + str(mean_transaction_size))

def getSummary(filename):

    data = pd.read_csv(os.getcwd() + "/../data/csv/" + filename)
    summary = data.describe()
    summary = summary.transpose()
    plot = plt.subplot(111, frame_on=False)

    #remove axis
    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False) 

    #create the table plot and position it in the upper left corner
    table(plot, summary,loc='upper right')

    #save the plot as a png file
    filename.replace(".csv", ".png")
    plt.savefig(os.getcwd() + "/../data/summaries/" + filename)


def getAllsummaries():
    for filename in os.listdir(os.getcwd() + "/../data/csv/"):
    # For each json file
        if "categories" in filename:
            getSummary(filename)

if __name__ == "__main__":

    if(len(sys.argv) == 2):
        if(sys.argv[1] =="all"):
            getAllsummaries()
            getBusinessCategoriesMetrics()
        elif(sys.argv[1] =="summary"):
            getAllsummaries()
        elif(sys.argv[1] =="transactions"):
            getBusinessCategoriesMetrics()
        elif (sys.argv[1] == "itemsets"):
            getFreqItemsetsCurves()
        else:
            raise ValueError("Cet argument n'existe pas... encore !")