# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.decomposition import PCA
from data_loader import load_data


CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness-extended.csv"
DATA_DIR = "results"


def simplePCA():
    data_x, data_y, c, number_of_features = load_data(effective_non_effective = True,
                                                      coverage = True,
                                                      grano_test = True,
                                                      grano_production = True,
                                                      my_test = True,
                                                      my_production = True,
                                                      scale = True)
    pca = PCA(n_components = 10) #(n_components=number_of_features)
    #principalComponents = pca.fit_transform(data_x)
    pca.fit(data_x)
    #data_x = pd.DataFrame(data = principalComponents, columns=c)
    df = pd.DataFrame(pca.components_, columns = c)
    #print(df)

    df = df.round(3)

    exp = pca.explained_variance_ratio_
    for i, j in df.iterrows():
        print("Component {}, Explains {} of the variance".format(i,exp[i]))
        print(j[abs(j) > 0.1])
        print("--------------------------------")


    df.to_csv(r'PCA.csv')


def main():
	s = simplePCA()



if __name__ == '__main__':
    main()
