import numpy as np
import pandas as pd


import matplotlib
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold,\
    cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    mean_absolute_error, make_scorer, brier_score_loss, roc_curve

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')



CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness.csv"
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





grano_test_data =






grano_production_data = ['csm_CDSBP',
                         'csm_CC',
                         'csm_FD',
                         'csm_Blob',
                         'csm_SC',
                         'csm_MC',
                         'csm_LM',
                         'csm_FE',
                         'LOC_prod',
                         'HALSTEAD_prod',
                         'RFC_prod',
                         'CBO_prod',
                         'MPC_prod',
                         'IFC_prod',
                         'DAC_prod',
                         'DAC2_prod',
                         'LCOM1_prod',
                         'LCOM2_prod',
                         'LCOM3_prod',
                         'LCOM4_prod',
                         'CONNECTIVITY_prod',
                         'LCOM5_prod',
                         'COH_prod',
                         'TCC_prod',
                         'LCC_prod',
                         'ICH_prod',
                         'WMC_prod',
                         'NOA_prod',
                         'NOPA_prod',
                         'NOP_prod',
                         'McCABE_prod',
                         'BUSWEIMER_prod',
                         'prod_readability']


my_test_data = ['No. Methods',
                'Vocabulary',
                'Word',
                'Special',
                'Non Whithe Characters',
                'No. Method Invoctions',
                'AST size',
                'Max Depth',
                'Avg Depth',
                'Deg^2',
                'Deg^3',
                'Deg',
                'Deg^-1',
                'Deg^-2',
                ]




my_general = ['No. Methods',
              'Vocabulary',
              'Word',
              'Special',
              'Non Whithe Characters',
              'No. Method Invoctions',
              'AST size',
              'Max Depth',
              'Avg Depth',
              'Deg^2',
              'Deg^3',
              'Deg',
              'Deg^-1',
              'Deg^-2',
                ]

]
my_production_data = [

]

                'Deg^2_prod','Deg^3_prod',
                'Deg_prod','Deg^-1_prod','Deg^-2_prod',
                'Decendent', 'Decendent_prod',

                         'Avg Depth^(-2)', 'Avg Depth^(-2)_prod',
                         'Avg Depth^(-1)', 'Avg Depth^(-1)_prod',
                         'Avg Depth^2', 'Avg Depth^2_prod',
                        'Avg Depth^3', 'Avg Depth^3_prod',

                'DegPerm',

                 'Dexterity', 'No. Expressions', 'No. Try', 'No. Catch',
               'No. Loop', 'No. Break', 'No. Continue', 'No. Conditions', 'No. Else', 'Bad API',
               'Junit', 'Hamcrest', 'Mockito', 'No. Methods_prod', 'Vocabulary_prod', 'Word_prod',
               'Special_prod', 'Non Whithe Characters_prod', 'No. Method Invoctions_prod', 'AST size_prod',
               'Max Depth_prod', 'Avg Depth_prod', 'DegPerm_prod', 'Dexterity_prod',
               'No. Expressions_prod', 'No. Try_prod', 'No. Catch_prod', 'No. Loop_prod', 'No. Break_prod',
               'No. Continue_prod', 'No. Conditions_prod', 'No. Else_prod',
               'Strings', 'Strings_prod', 'Numeric Literals', 'Numeric Literals_prod',
               'Comments' , 'Comments_prod', 'No. Field Access' , 'No. Field Access_prod',
               'No. Primitives' , 'No. Primitives_prod',
                'No. &&', 'No. &&_prod',  'No. ||', 'No. ||_prod', 'No. Ternary', 'No. Ternary_prod']










def label_rename1 (row):
    return row['path_test'].split('/')[len(row['path_test'].split('/')) - 1].split('.')[0]

def label_rename2 (row):
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

    d = {'TestClassName' : 'ClassName',
         'Vocabulary' : 'Vocabulary_prod',
         'Word' : 'Word_prod',
         'Non Whithe Characters' : 'Non Whithe Characters_prod',
         'No. Methods' : 'No. Methods_prod',
         'Special' : 'Special_prod',
         'No. Method Invoctions' : 'No. Method Invoctions_prod',
         'AST size' : 'AST size_prod',
         'Max Depth' : 'Max Depth_prod',
         'Deg2' : 'Deg2_prod',
         'DegPerm' : 'DegPerm_prod',
         'No. Break' : 'No. Break_prod',
         'No. Continue' : 'No. Continue_prod',
         'Avg Depth' : 'Avg Depth_prod', 'Dexterity' : 'Dexterity_prod',
         'No. Expressions' : 'No. Expressions_prod',
         'No. Try' : 'No. Try_prod',
         'No. Catch' : 'No. Catch_prod',
         'No. Loop' : 'No. Loop_prod',
         'No. Conditions' : 'No. Conditions_prod',
         'No. Else' : 'No. Else_prod',
         'Strings' : 'Strings_prod',
         'Numeric Literals':'Numeric Literals_prod',
         'Comments' : 'Comments_prod',
         'No. Field Access' : 'No. Field Access_prod',
         'No. Primitives' : 'No. Primitives_prod',
         'Avg Depth Squared' : 'Avg Depth Squared_prod',
     'No. &&' : 'No. &&_prod',  'No. ||' : 'No. ||_prod', 'No. Ternary': 'No. Ternary_prod'}


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



def delete_by_values(lst, values):
    values_as_set = set(values)
    return [ x for x in lst if x not in values_as_set ]

def pick_data(coverage, grano_test,
              grano_production, my_test, my_production, except):
              res = []
              if coverage:
                  res += line_coverage
              if grano_test:
                  res += grano_test_data
              if grano_production:
                  res += grano_production_data
              if my_test:
                  res += my_test_data
              if my_production:
                  res += my_test_data
              return delete_by_values(res, except)


def load_data(effective_non_effective = False,
              coverage = False,
              grano_test = False,
              grano_production = False,
              my_test = False,
              my_production = False,
              except = []):
                  frame = load_frame()
                  if effective_non_effective:
                      frame = load_quartile(frame)
                  columns = pick_data(coverage, grano_test, grano_production, my_test, my_production, except)
                  data_x = frame[columns]
                  data_y = pd.concat([frame.mutation], axis = 1)

                  return data_x, data_y, columns, len(columns)
