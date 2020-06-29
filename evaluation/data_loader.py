import numpy as np
import pandas as pd


#import matplotlib
#from matplotlib import pyplot as plt
#from sklearn.externals import joblib
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



#from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold,\
#    cross_validate, RandomizedSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
#    mean_absolute_error, make_scorer, brier_score_loss, roc_curve

#from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


from sklearn.utils import shuffle
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

#from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

#matplotlib.use('Agg')



CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness-extended.csv"
DATA_DIR = "results"



line_coverage = ['line_coverage']


grano_general = ['LOC',
                 'HALSTEAD',
                 'RFC',
                 'CBO',
                 'MPC',
                 'IFC',
                 'DAC',
                 'DAC2',
                 'LCOM1',
                 'LCOM2',
                 'LCOM3',
                 'LCOM4',
                 'CONNECTIVITY',
                 'LCOM5',
                 'COH',
                 'TCC',
                 'LCC',
                 'ICH',
                 'WMC',
                 'NOA',
                 'NOPA',
                 'NOP',
                 'McCABE',
                 'BUSWEIMER']


test_smells = ['isAssertionRoulette',
               'isEagerTest',
               'isLazyTest',
               'isMysteryGuest',
               'isSensitiveEquality',
               'isResourceOptimism',
               'isForTestersOnly',
               'isIndirectTesting']


code_smells = ['csm_CDSBP',
               'csm_CC',
               'csm_FD',
               'csm_Blob',
               'csm_SC',
               'csm_MC',
               'csm_LM',
               'csm_FE']


my_general = ['No. Methods',
              'Vocabulary',
              'Word',
              'Special',
              'Non Whithe Characters',
              'No. Method Invoctions',
              'AST size',
              'Max Depth',
              'Deg^2',
              'Deg^3',
              'Deg',
              'Deg^-1',
              'Deg^-2',
              'Decendent',
              'Avg Depth^(-2)',
              'Avg Depth^(-1)',
              'Avg Depth',
              'Avg Depth^2',
              'Avg Depth^3',
              'Avg Depth^3_prod',
              'DegPerm',
              'Dexterity',
              'No. Expressions',
              'No. Try',
              'No. Catch',
              'No. Loop',
              'No. Break',
              'No. Continue',
              'No. Conditions',
              'No. Else',
              'Strings',
              'Numeric Literals',
              'Comments',
              'No. Field Access',
              'No. Primitives' ,
              'No. &&',
              'No. ||',
              'No. Ternary']


test_frameworks = ['Bad API',
                   'Junit',
                   'Hamcrest',
                   'Mockito']


grano_production_data = [(factor + "_production") for factor in grano_general] + code_smells + ['prod_readability']
grano_test_data = [(factor + "_test") for factor in grano_general] + test_smells + ['test_readability']
my_test_data = my_general + test_frameworks
my_production_data = [(factor + "_production") for factor in my_general]


def label_rename1(row):
    return row['path_test'].split('/')[len(row['path_test'].split('/')) - 1].split('.')[0]

def label_rename2(row):
    return row['path_src'].split('/')[len(row['path_src'].split('/')) - 1].split('.')[0]

def load_quartile(frame):
    low, high = frame.mutation.quantile([0.25,0.75])
    frame_low = frame.query('mutation<{low}'.format(low=low))
    frame_high = frame.query('mutation>{high}'.format(high=high))
    frame_low['mutation'] = 0
    frame_high['mutation'] = 1
    frame = pd.concat([frame_low, frame_high], ignore_index=True)
    frame = frame.sample(frac=1).reset_index(drop=True)
    return frame;


def load_frame():

    d = dict(zip(my_general,[(factor + "_production") for factor in my_general]))
    d['TestClassName'] = 'ClassName'

    frame1 = pd.read_csv(CSV_PATH, sep=",")
    frame1 = frame1.sample(frac=1).reset_index(drop=True)
    frame1['TestClassName'] = frame1.apply(lambda row: label_rename1(row), axis=1)
    frame1['ClassName'] = frame1.apply(lambda row: label_rename2(row), axis=1)

    frame2 = pd.read_csv(CSV_MINER_PATH, sep=',')

    frame3 = pd.read_csv(CSV_MINER_PATH, sep=',')
    frame3 = frame3.rename(columns = d)
    frame3 = frame3.drop(['Bad API', 'Junit', 'Hamcrest', 'Mockito', 'Nº','Project'], axis=1)


    frame = pd.merge(frame1, frame2, on='TestClassName')
    frame = pd.merge(frame, frame3, on='ClassName')
    frame = frame.drop(['project', 'module', 'path_test','test_name','path_src',
                        'commit', 'class_name'], axis=1)
    frame = frame.sample(frac=1).reset_index(drop=True)
    frame = frame.dropna()
    return frame



def main():
    s = load_frame()
    print(s)


if __name__ == '__main__':
    main()
