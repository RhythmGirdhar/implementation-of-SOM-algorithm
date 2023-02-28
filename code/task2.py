__author__ = 'Rhythm Girdhar'
__email__ = 'rgirdhar@usc.edu'

from collections import defaultdict
import copy
import csv
import math
from operator import add
import sys
import json
import pyspark
import time
import itertools
from itertools import combinations
from pyspark.context import SparkContext


def frequent_item_set_k1(partition, all_item_set, part_support):
    freq_itemsets = [item for item in all_item_set if sum(1 for basket in partition if item in basket) >= part_support]
    freq_itemsets.sort()
    return freq_itemsets

def check_monotonicity(candidate, candidate_combo, k):
    candidate_set = set(map(frozenset, candidate))
    res_set = set(map(frozenset, itertools.combinations(candidate_combo, k-1)))
    
    return res_set.issubset(candidate_set)

def create_combinations(candidate, k):
    final_set = set()
    final_result = []
    if k == 2:
        for l in candidate:
            final_set.add(l)
        res = itertools.combinations(final_set, k)
        res = list(map(lambda x: tuple(sorted(x)), res))
        return res
    else:
        for l in candidate:
            for i in l:
                final_set.add(i)  
        res = itertools.combinations(final_set, k)
        res = list(map(lambda x: tuple(sorted(x)), res))
        for r in res:
            if check_monotonicity(candidate, r, k):
                final_result.append(r)
        return final_result


def get_frequent_item_sets(partition, candidate_set, part_support): 
    freq_itemsets = {item for item in candidate_set if sum(1 for basket in partition if set(item).issubset(basket)) >= part_support}
    return freq_itemsets

def apriori(partition, whole_support):
    partition = list(partition)
    all_item_set = set().union(*partition)

    part_support = math.ceil(whole_support * len(partition) / total_baskets)

    # for k = 1
    candidate_L1 = frequent_item_set_k1(partition, all_item_set, part_support)
    candidate_list = [[(i,) for i in candidate_L1]]

    candidate_set = create_combinations(candidate_L1, 2)

    k = 2
    while candidate_set:
        candidate_L = get_frequent_item_sets(partition, candidate_set, part_support)
        candidate_list.append(list(candidate_L))
        candidate_set = create_combinations(candidate_L, k+1)
        k += 1

    return candidate_list


def count_all(partition, candidates):
    candidate_count = defaultdict(int)
    partition_set = set(partition)
    for candidate in candidates:
        if set(candidate).issubset(partition_set):
            candidate_count[tuple(candidate)] += 1

    return list(candidate_count.items())


def format_output(result, f):
    dictionary_1 = {i: [] for i in range(1, max(map(len, result)) + 1)}
    for i in result:
        dictionary_1[len(i)].append(i)

    dic_sort = {i: [] for i in range(1, max(map(len, result)) + 1)}
    # SORT LEXICOGRAPHICALLY
    for i in dictionary_1:
        for j in dictionary_1[i]:
            set1 = sorted(set(p for p in j))
            dic_sort[i].append(set1)

    dic_sort1 = {i: [] for i in range(1, max(map(len, result)) + 1)}
    for i in dic_sort:
        dic_sort1[i] = sorted(dic_sort[i], key=lambda x: tuple(x[j] for j in range(i)))

    for i in dic_sort1:
        s = ""
        for j in dic_sort1[i][:-1]:
            if i == 1:
                s = s + "('" + j[0] + "')" + ','
            else:
                s = s + str(tuple(j)) + ','
        if i == 1:
            s = s + "('" + dic_sort1[i][-1][0] + "')\n\n"
        else:
            s = s + str(tuple((dic_sort1[i][-1]))) + "\n\n"

        f.write(s)

def write(output_file, mr_phase_1, mr_phase_2):
    with open (output_file, "w") as f:
        f.write("Candidates:\n")
        format_output(mr_phase_1, f)
        f.write("Frequent Itemsets:\n")
        format_output(mr_phase_2, f)


# def write_csv(output_file, )
if __name__ == "__main__":

    # Take Command-Line Arguments from the user

    # filter_threshold = int(sys.argv[1])
    # support = int(sys.argv[2])
    # input_file = sys.argv[3]
    # output_file = sys.argv[4]

    # Hard-coded Values for Testing Purposes

    filter_threshold = 20
    support = 4
    input_file = "../data/small1.csv"
    output_file = "../result/task1_1.txt"

    sc = SparkContext()

    start = time.time()
    data_RDD = sc.textFile(input_file).map(lambda x: x.split(','))
    header = data_RDD.first()
    # data_RDD = data_RDD.filter(lambda x : x != header).map(lambda x: (x[0][1:-5] + x[0][-3:-1] + "-" + x[1][1:-1].lstrip("0"), x[5][1:-1].lstrip("0")))

    data_RDD = data_RDD.filter(lambda x : x != header)

    with open('customer_product.csv', 'w') as output:
        writer = csv.writer(output)
        writer.writerow(["DATE-CUSTOMER_ID", "PRODUCT_ID"])
        for i in data_RDD.collect():
            writer.writerow(i)
            
    # basket_RDD = data_RDD.groupByKey().map(lambda x : set(x[1])).filter(lambda x: len(x) > filter_threshold)

    basket_RDD = data_RDD.groupByKey().map(lambda x : set(x[1]))

    total_baskets = basket_RDD.count()

    # Implement SON Algorithm

    phase_1 = basket_RDD.mapPartitions(lambda partition: apriori(partition, support)).map(lambda x: (tuple(sorted(x)))).flatMap(lambda x: x).distinct().collect()

    phase_2 = basket_RDD.flatMap(lambda partition: count_all(partition, phase_1)).reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support).map(lambda x: x[0]).sortBy(lambda pairs: (len(pairs), pairs)).collect()

    write(output_file, phase_1, phase_2)
    print("Duration: ", (time.time()) - start)