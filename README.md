# YelpAssociationMining

Association mining project on the [Yelp Dataset](https://www.yelp.com/dataset).

The goal of this project was to perform an association mining case study on Yelp business categories and to compare the performance of different mining algorithms.
Three association mining algorithms are compared: Apriori [Agrawal and  Srikant, 1994], FP-Growth [Han et al., 2000] and Eclat [Zaki, 2000].
For Apriori and FP-Growth, we use the [mlxtend](https://github.com/rasbt/mlxtend) implementation. For Eclat, we use the [PyECLAT](https://github.com/jeffrichardchemistry/pyECLAT) implementation.
Performance comparison metric is the execution time, measured for different minsup.

The files in data/ and results/ folders are just sample files to illustrate the structure of the project. In order to use the code, you have to download and extract the Yelp dataset into the data folder.

### References:
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
