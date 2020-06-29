# -*- coding: utf-8 -*-

#import numpy as np
#import pandas as pd


#import matplotlib
#from matplotlib import pyplot as plt
#from sklearn.externals import joblib
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



#from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold, \
#    cross_validate, RandomizedSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
#    mean_absolute_error, make_scorer, brier_score_loss, roc_curve

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.decomposition import PCA


#from sklearn.utils import shuffle
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

#from tensorflow import keras

#import warnings
#warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"s

#matplotlib.use('Agg')

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
