# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


import matplotlib
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold, \
    cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    mean_absolute_error, make_scorer, brier_score_loss, roc_curve

from sklearn.preprocessing import OneHotEncoder

from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"s

matplotlib.use('Agg')



__author__ = "Dor Ma'ayan"
__email__ = "grano@ifi.uzh.ch"
__license__ = "MIT"


CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness.csv"
DATA_DIR = "results"


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
    'AST size' : 'AST size_prod', 'Max Depth' : 'Max Depth_prod',
         'Deg2' : 'Deg2_prod',
         'DegPerm' : 'DegPerm_prod',
         'No. Break' : 'No. Break_prod',
         'No. Continue' : 'No. Continue_prod',
     'Avg Depth' : 'Avg Depth_prod', 'Dexterity' : 'Dexterity_prod',
    'No. Expressions' : 'No. Expressions_prod', 'No. Try' : 'No. Try_prod', 'No. Catch' : 'No. Catch_prod',
     'No. Loop' : 'No. Loop_prod', 'No. Conditions' : 'No. Conditions_prod', 'No. Else' : 'No. Else_prod'}
    
    
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


def load_quartile(frame):
    low, high = frame.mutation.quantile([0.25,0.75])
    frame_low = frame.query('mutation<{low}'.format(low=low))
    frame_high = frame.query('mutation>{high}'.format(high=high))
    frame_low['mutation'] = 0
    frame_high['mutation'] = 1
    frame = pd.concat([frame_low, frame_high], ignore_index=True)
    frame = frame.sample(frac=1).reset_index(drop=True)
    return frame;



def load_all_data(frame):
    columns = ['isAssertionRoulette',
       'isEagerTest', 'isLazyTest', 'isMysteryGuest',
       'isSensitiveEquality', 'isResourceOptimism', 'isForTestersOnly',
       'isIndirectTesting', 'LOC_prod', 'HALSTEAD_prod', 'RFC_prod',
       'CBO_prod', 'MPC_prod', 'IFC_prod', 'DAC_prod', 'DAC2_prod',
       'LCOM1_prod', 'LCOM2_prod', 'LCOM3_prod', 'LCOM4_prod',
       'CONNECTIVITY_prod', 'LCOM5_prod', 'COH_prod', 'TCC_prod',
       'LCC_prod', 'ICH_prod', 'WMC_prod', 'NOA_prod', 'NOPA_prod',
       'NOP_prod', 'McCABE_prod', 'BUSWEIMER_prod', 'LOC_test',
       'HALSTEAD_test', 'RFC_test', 'CBO_test', 'MPC_test', 'IFC_test',
       'DAC_test', 'DAC2_test', 'LCOM1_test', 'LCOM2_test', 'LCOM3_test',
       'LCOM4_test', 'CONNECTIVITY_test', 'LCOM5_test', 'COH_test',
       'TCC_test', 'LCC_test', 'ICH_test', 'WMC_test', 'NOA_test',
       'NOPA_test', 'NOP_test', 'McCABE_test', 'BUSWEIMER_test',
       'csm_CDSBP', 'csm_CC', 'csm_FD', 'csm_Blob', 'csm_SC', 'csm_MC',
       'csm_LM', 'csm_FE', 'prod_readability', 'test_readability', 'No. Methods', 'Vocabulary', 'Word',
               'Special', 'Non Whithe Characters', 'No. Method Invoctions', 'AST size', 'Max Depth',
               'Avg Depth', 'Deg2', 'DegPerm', 'Dexterity', 'No. Expressions', 'No. Try', 'No. Catch',
               'No. Loop', 'No. Break', 'No. Continue', 'No. Conditions', 'No. Else', 'Bad API',
               'Junit', 'Hamcrest', 'Mockito', 'No. Methods_prod', 'Vocabulary_prod', 'Word_prod',
               'Special_prod', 'Non Whithe Characters_prod', 'No. Method Invoctions_prod', 'AST size_prod',
               'Max Depth_prod', 'Avg Depth_prod', 'Deg2_prod', 'DegPerm_prod', 'Dexterity_prod',
               'No. Expressions_prod', 'No. Try_prod', 'No. Catch_prod', 'No. Loop_prod', 'No. Break_prod',
               'No. Continue_prod', 'No. Conditions_prod', 'No. Else_prod']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)

def load_all_data_dynamic(frame):
    columns = ['line_coverage', 'isAssertionRoulette',
       'isEagerTest', 'isLazyTest', 'isMysteryGuest',
       'isSensitiveEquality', 'isResourceOptimism', 'isForTestersOnly',
       'isIndirectTesting', 'LOC_prod', 'HALSTEAD_prod', 'RFC_prod',
       'CBO_prod', 'MPC_prod', 'IFC_prod', 'DAC_prod', 'DAC2_prod',
       'LCOM1_prod', 'LCOM2_prod', 'LCOM3_prod', 'LCOM4_prod',
       'CONNECTIVITY_prod', 'LCOM5_prod', 'COH_prod', 'TCC_prod',
       'LCC_prod', 'ICH_prod', 'WMC_prod', 'NOA_prod', 'NOPA_prod',
       'NOP_prod', 'McCABE_prod', 'BUSWEIMER_prod', 'LOC_test',
       'HALSTEAD_test', 'RFC_test', 'CBO_test', 'MPC_test', 'IFC_test',
       'DAC_test', 'DAC2_test', 'LCOM1_test', 'LCOM2_test', 'LCOM3_test',
       'LCOM4_test', 'CONNECTIVITY_test', 'LCOM5_test', 'COH_test',
       'TCC_test', 'LCC_test', 'ICH_test', 'WMC_test', 'NOA_test',
       'NOPA_test', 'NOP_test', 'McCABE_test', 'BUSWEIMER_test',
       'csm_CDSBP', 'csm_CC', 'csm_FD', 'csm_Blob', 'csm_SC', 'csm_MC',
       'csm_LM', 'csm_FE', 'prod_readability', 'test_readability', 'No. Methods', 'Vocabulary', 'Word',
               'Special', 'Non Whithe Characters', 'No. Method Invoctions', 'AST size', 'Max Depth',
               'Avg Depth', 'Deg2', 'DegPerm', 'Dexterity', 'No. Expressions', 'No. Try', 'No. Catch',
               'No. Loop', 'No. Break', 'No. Continue', 'No. Conditions', 'No. Else', 'Bad API',
               'Junit', 'Hamcrest', 'Mockito', 'No. Methods_prod', 'Vocabulary_prod', 'Word_prod',
               'Special_prod', 'Non Whithe Characters_prod', 'No. Method Invoctions_prod', 'AST size_prod',
               'Max Depth_prod', 'Avg Depth_prod', 'Deg2_prod', 'DegPerm_prod', 'Dexterity_prod',
               'No. Expressions_prod', 'No. Try_prod', 'No. Catch_prod', 'No. Loop_prod', 'No. Break_prod',
               'No. Continue_prod', 'No. Conditions_prod', 'No. Else_prod']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)


def load_all_their_data(frame):
    columns = ['isAssertionRoulette',
       'isEagerTest', 'isLazyTest', 'isMysteryGuest',
       'isSensitiveEquality', 'isResourceOptimism', 'isForTestersOnly',
       'isIndirectTesting', 'LOC_prod', 'HALSTEAD_prod', 'RFC_prod',
       'CBO_prod', 'MPC_prod', 'IFC_prod', 'DAC_prod', 'DAC2_prod',
       'LCOM1_prod', 'LCOM2_prod', 'LCOM3_prod', 'LCOM4_prod',
       'CONNECTIVITY_prod', 'LCOM5_prod', 'COH_prod', 'TCC_prod',
       'LCC_prod', 'ICH_prod', 'WMC_prod', 'NOA_prod', 'NOPA_prod',
       'NOP_prod', 'McCABE_prod', 'BUSWEIMER_prod', 'LOC_test',
       'HALSTEAD_test', 'RFC_test', 'CBO_test', 'MPC_test', 'IFC_test',
       'DAC_test', 'DAC2_test', 'LCOM1_test', 'LCOM2_test', 'LCOM3_test',
       'LCOM4_test', 'CONNECTIVITY_test', 'LCOM5_test', 'COH_test',
       'TCC_test', 'LCC_test', 'ICH_test', 'WMC_test', 'NOA_test',
       'NOPA_test', 'NOP_test', 'McCABE_test', 'BUSWEIMER_test',
       'csm_CDSBP', 'csm_CC', 'csm_FD', 'csm_Blob', 'csm_SC', 'csm_MC',
       'csm_LM', 'csm_FE', 'prod_readability', 'test_readability']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)

def load_all_their_test_data(frame):
    columns = ['isAssertionRoulette',
       'isEagerTest', 'isLazyTest', 'isMysteryGuest',
       'isSensitiveEquality', 'isResourceOptimism', 'isForTestersOnly',
       'isIndirectTesting','LOC_test',
       'HALSTEAD_test', 'RFC_test', 'CBO_test', 'MPC_test', 'IFC_test',
       'DAC_test', 'DAC2_test', 'LCOM1_test', 'LCOM2_test', 'LCOM3_test',
       'LCOM4_test', 'CONNECTIVITY_test', 'LCOM5_test', 'COH_test',
       'TCC_test', 'LCC_test', 'ICH_test', 'WMC_test', 'NOA_test',
       'NOPA_test', 'NOP_test', 'McCABE_test', 'BUSWEIMER_test', 'test_readability']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)

def load_all_test_data(frame):
    columns = ['isAssertionRoulette',
       'isEagerTest', 'isLazyTest', 'isMysteryGuest',
       'isSensitiveEquality', 'isResourceOptimism', 'isForTestersOnly',
       'isIndirectTesting', 'LOC_test',
       'HALSTEAD_test', 'RFC_test', 'CBO_test', 'MPC_test', 'IFC_test',
       'DAC_test', 'DAC2_test', 'LCOM1_test', 'LCOM2_test', 'LCOM3_test',
       'LCOM4_test', 'CONNECTIVITY_test', 'LCOM5_test', 'COH_test',
       'TCC_test', 'LCC_test', 'ICH_test', 'WMC_test', 'NOA_test',
       'NOPA_test', 'NOP_test', 'McCABE_test', 'BUSWEIMER_test',
       'test_readability', 'No. Methods', 'Vocabulary', 'Word',
               'Special', 'Non Whithe Characters', 'No. Method Invoctions', 'AST size', 'Max Depth',
               'Avg Depth', 'Deg2', 'DegPerm', 'Dexterity', 'No. Expressions', 'No. Try', 'No. Catch',
               'No. Loop', 'No. Break', 'No. Continue', 'No. Conditions', 'No. Else', 'Bad API',
               'Junit', 'Hamcrest', 'Mockito']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)


def load_all_production_data(frame):
    columns = ['LOC_prod', 'HALSTEAD_prod', 'RFC_prod',
       'CBO_prod', 'MPC_prod', 'IFC_prod', 'DAC_prod', 'DAC2_prod',
       'LCOM1_prod', 'LCOM2_prod', 'LCOM3_prod', 'LCOM4_prod',
       'CONNECTIVITY_prod', 'LCOM5_prod', 'COH_prod', 'TCC_prod',
       'LCC_prod', 'ICH_prod', 'WMC_prod', 'NOA_prod', 'NOPA_prod',
       'NOP_prod', 'McCABE_prod', 'BUSWEIMER_prod',
       'csm_CDSBP', 'csm_CC', 'csm_FD', 'csm_Blob', 'csm_SC', 'csm_MC',
       'csm_LM', 'csm_FE', 'prod_readability', 'No. Methods_prod', 'Vocabulary_prod', 'Word_prod',
               'Special_prod', 'Non Whithe Characters_prod', 'No. Method Invoctions_prod', 'AST size_prod',
               'Max Depth_prod', 'Avg Depth_prod', 'Deg2_prod', 'DegPerm_prod', 'Dexterity_prod',
               'No. Expressions_prod', 'No. Try_prod', 'No. Catch_prod', 'No. Loop_prod', 'No. Break_prod',
               'No. Continue_prod', 'No. Conditions_prod', 'No. Else_prod']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)


def load_all_their_production_data(frame):
    columns = ['LOC_prod', 'HALSTEAD_prod', 'RFC_prod',
       'CBO_prod', 'MPC_prod', 'IFC_prod', 'DAC_prod', 'DAC2_prod',
       'LCOM1_prod', 'LCOM2_prod', 'LCOM3_prod', 'LCOM4_prod',
       'CONNECTIVITY_prod', 'LCOM5_prod', 'COH_prod', 'TCC_prod',
       'LCC_prod', 'ICH_prod', 'WMC_prod', 'NOA_prod', 'NOPA_prod',
       'NOP_prod', 'McCABE_prod', 'BUSWEIMER_prod', 'csm_CDSBP', 'csm_CC', 'csm_FD', 'csm_Blob', 'csm_SC', 'csm_MC',
       'csm_LM', 'csm_FE', 'prod_readability']

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)

    data_x = frame[columns].round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)

def get_scoring():
    """Returns the scores to evaluate the model"""
    return dict(accuracy=make_scorer(accuracy_score),
                precision=make_scorer(precision_score),
                recall=make_scorer(recall_score),
                f1_score=make_scorer(f1_score),
                roc_auc_scorer=make_scorer(roc_auc_score),
                mean_absolute_error=make_scorer(mean_absolute_error),
                brier_score=make_scorer(brier_score_loss))


def create_model():
  model = keras.Sequential()
  model.add(keras.layers.Dense(number_of_features, activation='relu', input_dim=number_of_features))
  model.add(keras.layers.Dense(100, activation='relu'))
  model.add(keras.layers.Dense(20, activation='relu'))
  model.add(keras.layers.Dense(1))

  model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae','mse'])
  return model



def main():
  frame = load_frame()
  data_x, data_y, number_of_features = load_all_data_dynamic(frame)
  scaler = StandardScaler()
  scaler.fit(data_x)
  data_x = scaler.transform(data_x)

  model = KerasClassifier(build_fn=create_model, verbose=0, epochs=10)

  results = cross_validate(estimator=model,
    cv=outer_cv,
    X=data_x,
    y=data_y,
    scoring=('mae', 'mse'),
    return_train_score=False,
    verbose=1,
    n_jobs=-1)

  print(results)
