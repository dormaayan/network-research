import pandas as pd
from sklearn.decomposition import PCA
from data_loader import load_data, get_category


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
    pca = PCA()
    pca.fit(data_x)
    df = pd.DataFrame(pca.components_, columns = c)

    exp = pca.explained_variance_ratio_
    for i, j in df.iterrows():
        print("Component {}, Explains {} of the variance".format(i,exp[i]))
        print(j[abs(j) > 0.1])
        print("--------------------------------")

    res = {}
    for i, j in df.iteritems():
        print("Feature: {} \n".format(i))
        print((abs(j) * exp).sum())
        res[i] = (abs(j) * exp).sum()

    sorted_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    for i in sorted_res.keys():
        print('category: {}, implication: {}'.format(get_category(i), sorted_res[i]))
    df.to_csv(r'PCA.csv')


def main():
    simplePCA()

if __name__ == '__main__':
    main()
