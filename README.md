# YelpAssociationMining

Association mining project on the [Yelp Dataset](https://www.yelp.com/dataset).

The goal of this project was to perform an association mining case study on Yelp business categories and to compare the performance of different mining algorithms.

For the case study on Yelp businesses, there are two goals: identify frequent business categories and most popular business categories. This is done in general, and then for the Montreal and Philadelphia markets specifically. Popularity is based on user reviews (>=3 stars and above).

For the comparative study on algorithmic performance, three association mining algorithms are evaluated: Apriori [Agrawal and  Srikant, 1994], FP-Growth [Han et al., 2000] and Eclat [Zaki, 2000].
For Apriori and FP-Growth, we use the [mlxtend](https://github.com/rasbt/mlxtend) implementation. For Eclat, we use the [PyECLAT](https://github.com/jeffrichardchemistry/pyECLAT) implementation.
Performance comparison metric is the execution time, measured for different minsup.

## Scripts:

Note: the files in data/ and results/ folders are just sample files to illustrate the structure of the project. In order to use the scripts below, you have to download and extract the Yelp dataset into the data/ folder.

* extract_data.py: save json files to csv format
* preprocess.py: form dataframes to make way for association mining (remove empty lines, join DFs, filter for reviews with >=3 stars, etc.)
* catacteristics.py: summarize some frequent items characteristics for the Yelp dataset (e.g. max/mean transaction size, frequent itemset curves for different values of minsup, etc.)
* frequent_categories.py: identify frequent categories/associations in all businesses
* frequent_restaurant_categories.py: identify frequent categories/associations for restaurant businesses
* frequent_user_categories.py: identify frequent categories/associations for user-preferred businesses (based on reviews)
* measure_exec_time.py measure execution time on the Yelp dataset for all 3 association mining algorithms.

## References:
* Agrawal, R. and Srikant, R. (1994).
Fast algorithms for mining association rules in large databases.
In Proceedings of the 20th International Conference on Very Large
Data Bases, VLDB ’94, p. 487–499., San Francisco, CA, USA.
Morgan Kaufmann Publishers Inc.
* Han, J., Pei, J. and Yin, Y. (2000).
Mining frequent patterns without candidate generation.
SIGMOD Rec., 29(2), 1–12.
http://dx.doi.org/10.1145/335191.335372.
* Zaki, M. (2000).
Scalable algorithms for association mining.
IEEE Transactions on Knowledge and Data Engineering, 12(3),
372–390.
http://dx.doi.org/10.1109/69.846291
