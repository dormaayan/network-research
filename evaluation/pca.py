# -*- coding: utf-8 -*-

import csv
import pandas as pd
from sklearn.decomposition import PCA
from data_loader import load_data, get_category


CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness-extended.csv"
DATA_DIR = "results"


def do_pca():
    data_x, data_y, columns, number_of_features = load_data(
        effective_non_effective=True, coverage=False, grano_test=True,
        grano_production=True, my_test=True, my_production=True, scale=True)
    pca = PCA()
    pca.fit(data_x)
    return pca, columns


def convert_pca(n_factors):
    data_x, data_y, columns, number_of_features = load_data(
        effective_non_effective=True, coverage=False, grano_test=True,
        grano_production=True, my_test=True, my_production=True, scale=True)
    pca = PCA()
    pca.fit_transform(data_x)
    return pca, columns

def analyze_componenets(n_factors):
    pca, columns = do_pca()
    df = pd.DataFrame(pca.components_, columns = columns)
    exp = pca.explained_variance_ratio_

    res = {}
    for i, j in df.iteritems():
        #print("Feature: {} \n".format(i))
        #print((abs(j) * exp).sum())
        res[i] = (abs(j) * exp).sum()

    #sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    
    sorted_res = sorted(res.items() , reverse=True, key=lambda x: x[1])


    count = 0
    top_factors = []
    for elem in sorted_res:
        if count < n_factors:
            
            a, b, c = get_category(elem[0])
            print('({}) category: {},{},{} - implication: {}'.format(count, a,b,c,elem[1]))
            top_factors.append(elem[0])
            count += 1

    return top_factors

def get_factors():
    pca, columns = do_pca()
    df = pd.DataFrame(pca.components_, columns=columns)

    exp = pca.explained_variance_ratio_
    #for i, j in df.iterrows():
        #print("Component {}, Explains {} of the variance".format(i,exp[i]))
        #print(j[abs(j) > 0.1])
        #print("--------------------------------")


def main():
    #simplePCA()

    get_factors()


if __name__ == '__main__':
    main()
