from collections import defaultdict
import copy
from doctest import testfile
import math
from operator import add
import sys
import json
import pyspark
import time
import itertools
from itertools import combinations
from pyspark.context import SparkContext


def apriori(partition, support):
    
    res = []
    
    baskets = list(partition)
    number_of_baskets = len(baskets)
    
    part_support = support * (number_of_baskets/total_baskets)
    
    C1 = {}
    for bask in baskets:
        for item in bask:
            if item not in C1:
                C1[item] = 1
            else:
                C1[item] += 1
    
    L1 = []
    for k,v in C1.items():
        if v>=part_support:
            L1.append(k)
    L1.sort()
    L1_tmp = [(i,) for i in L1]
    res.append(L1_tmp)
    
    L1_J = set(L1)
    
    C2 = {}
    for bask in baskets:
        common_singles = list(set(bask).union(L1_J))
        valid_combos = combinations(common_singles, 2)
        
        for vc in valid_combos:
            vc = tuple(sorted(vc))
            if vc not in C2:
                C2[vc] = 1
            else:
                C2[vc] += 1
    
    L2 = []
    for k,v in C2.items():
        if v>=part_support:
            L2.append(k)
    L2.sort()
    res.append(L2)
    
    L = L2
    k = 3

    
    while L:
        L_J = set()
        for item in L:
            L_J = L_J.union(set(item))

        Ck = {}

        for bask in baskets:
            common_singles = list(set(bask).intersection(L_J))
            valid_combos = combinations(common_singles, k)
            
            for vc in valid_combos:
                vc = tuple(sorted(vc))               
                if vc not in Ck:
                    Ck[vc] = 1
                else:
                    Ck[vc] += 1
    
        Lk = []
        for key,v in Ck.items():
            if v>=part_support:
                Lk.append(key)
        Lk.sort()
        res.append(Lk)

        k += 1
        L = Lk
    
    return res



def count_all(partition, candidates):
    candidate_count = defaultdict(int)
    partition_set = set(partition)
    for candidate in candidates:
        
        cand_set = set()
        if type(candidate) == str:
            cand_set.add(candidate)
        else:
            cand_set = set(candidate)
        
        
        if cand_set.issubset(partition_set):
            if type(candidate) == str:
                candidate_count[(candidate,)] += 1
            else:
                candidate_count[tuple(candidate)] += 1

    return candidate_count.items()


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
        f.truncate(f.tell()-1)

if __name__ == "__main__":
    # Take Command-Line Arguments from the user

    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    # Hard-coded Values for Testing Purposes

    # case_number = 1
    # support = 4
    # input_file = "../data/small1.csv"
    # output_file = "../result/task1_1.txt"

    sc = SparkContext()

    start = time.time()
    data_RDD = sc.textFile(input_file).map(lambda x: x.split(','))
    header = data_RDD.first()
    data_RDD = data_RDD.filter(lambda x : x != header)

    basket_RDD = None

    # market-basket model (user_id => business_id)
    if 1 == case_number:
        basket_RDD = data_RDD.groupByKey().map(lambda x : set(x[1]))
    
    # market-basket model (business_id => user_id)
    elif 2 == case_number:
        basket_RDD = data_RDD.map(lambda x : (x[1], x[0])).groupByKey().map(lambda x : set(x[1]))

    total_baskets = basket_RDD.count()
    
    # test_data = [{'m','c','b'}, {'m','c','b','n'}, {'m','p','b'}, {'c','b','j'}, {'m','p','j'}, {'c','j'}, {'m','c','b','j'}, {'b','c'}]
    # test_rdd = sc.parallelize(test_data)

    # Implement SON Algorithm

    
    
    phase_1 = basket_RDD.mapPartitions(lambda partition: apriori(partition, support)).map(lambda x: (tuple(sorted(x)))).flatMap(lambda x: x).distinct().collect()

    phase_2 = basket_RDD.flatMap(lambda partition: count_all(partition, phase_1)).reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support).map(lambda x: x[0]).sortBy(lambda pairs: (len(pairs), pairs)).collect()

    write(output_file, phase_1, phase_2)
    print("Duration: ", (time.time()) - start)

