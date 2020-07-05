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
        effective_non_effective=True, coverage=True, grano_test=True,
        grano_production=True, my_test=True, my_production=True, scale=True)
    pca = PCA()
    pca.fit(data_x)
    return pca, columns

def analyze_componenets(n_factors):
    pca, columns = do_pca()
    df = pd.DataFrame(pca.components_, columns = columns)
    exp = pca.explained_variance_ratio_

    res = {}
    for i, j in df.iteritems():
        print("Feature: {} \n".format(i))
        print((abs(j) * exp).sum())
        res[i] = (abs(j) * exp).sum()

    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    count = 0
    top_factors = []
    with open(r'PCA.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in sorted_res.keys():
            if count < n_factors:
                top_factors.append(i)
            print(i)
            a, b, c = get_category(i)
            print('category: {},{},{} - implication: {}'.format(a,b,c,sorted_res[i]))
            writer.writerow([i,sorted_res[i],a,b,c])

    return top_factors

def get_factors():
    pca, columns = do_pca()
    df = pd.DataFrame(pca.components_, columns=columns)

    exp = pca.explained_variance_ratio_
    for i, j in df.iterrows():
        print("Component {}, Explains {} of the variance".format(i,exp[i]))
        print(j[abs(j) > 0.1])
        print("--------------------------------")


def main():
    #simplePCA()

    #get_factors()
    analyze_componenets(60)


if __name__ == '__main__':
    main()
