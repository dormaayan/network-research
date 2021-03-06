# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import multiprocessing

import matplotlib
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold, \
    cross_validate
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

matplotlib.use('Agg')

import threading
lock = threading.Lock()

__author__ = "Dor Ma'ayan"
__email__ = "grano@ifi.uzh.ch"
__license__ = "MIT"


CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness.csv"
DATA_DIR = "results"

def label_rename (row):
    return row['path_test'].split('/')[len(row['path_test'].split('/')) - 1].split('.')[0]

def load_frame():
    frame1 = pd.read_csv(CSV_PATH, sep=",")
    frame1 = frame1.sample(frac=1).reset_index(drop=True)
    frame1['TestClassName'] = frame1.apply(lambda row: label_rename(row), axis=1)
    frame2 = pd.read_csv(CSV_MINER_PATH, sep=',')
    frame = pd.merge(frame1, frame2, on='TestClassName')
    frame = frame.drop(['project', 'module', 'path_test','test_name','path_src',
                        'class_name','TestClassName','commit','Nº','Project'], axis=1)
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
    columns = [frame.no_mutations, frame.line_coverage, frame.isAssertionRoulette, frame.isEagerTest, frame.isLazyTest,
frame.isMysteryGuest, frame.isSensitiveEquality, frame.isResourceOptimism, frame.isForTestersOnly,
frame.isIndirectTesting, frame.LOC_prod, frame.HALSTEAD_prod, frame.RFC_prod, frame.CBO_prod, frame.MPC_prod, frame.IFC_prod, frame.DAC_prod,frame.DAC2_prod, frame.LCOM1_prod, frame.LCOM2_prod,
frame.LCOM3_prod, frame.LCOM4_prod, frame.CONNECTIVITY_prod, frame.LCOM5_prod, frame.COH_prod, frame.TCC_prod,
frame.LCC_prod, frame.ICH_prod, frame.WMC_prod, frame.NOA_prod, frame.NOPA_prod, frame.NOP_prod,
frame.McCABE_prod, frame.BUSWEIMER_prod, frame.LOC_test, frame.HALSTEAD_test, frame.RFC_test, frame.CBO_test,
frame.MPC_test, frame.IFC_test, frame.DAC_test, frame.DAC2_test, frame.LCOM1_test, frame.LCOM2_test,
frame.LCOM3_test, frame.LCOM4_test, frame.CONNECTIVITY_test, frame.LCOM5_test, frame.COH_test, frame.TCC_test,
frame.LCC_test, frame.ICH_test, frame.WMC_test, frame.NOA_test, frame.NOPA_test, frame.NOP_test, frame.McCABE_test,
frame.BUSWEIMER_test, frame.csm_CDSBP, frame.csm_CC, frame.csm_FD, frame.csm_Blob, frame.csm_SC, frame.csm_MC,
frame.csm_LM, frame.csm_FE, frame.prod_readability, frame.test_readability]
    data_x = pd.concat(columns, axis = 1).round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)


def load_all_data_with_mine(frame):
    columns = [frame.no_mutations, frame.line_coverage, frame.isAssertionRoulette, frame.isEagerTest, frame.isLazyTest,
frame.isMysteryGuest, frame.isSensitiveEquality, frame.isResourceOptimism, frame.isForTestersOnly,
frame.isIndirectTesting, frame.LOC_prod, frame.HALSTEAD_prod, frame.RFC_prod, frame.CBO_prod, frame.MPC_prod, frame.IFC_prod, frame.DAC_prod,frame.DAC2_prod, frame.LCOM1_prod, frame.LCOM2_prod,
frame.LCOM3_prod, frame.LCOM4_prod, frame.CONNECTIVITY_prod, frame.LCOM5_prod, frame.COH_prod, frame.TCC_prod,
frame.LCC_prod, frame.ICH_prod, frame.WMC_prod, frame.NOA_prod, frame.NOPA_prod, frame.NOP_prod,
frame.McCABE_prod, frame.BUSWEIMER_prod, frame.LOC_test, frame.HALSTEAD_test, frame.RFC_test, frame.CBO_test,
frame.MPC_test, frame.IFC_test, frame.DAC_test, frame.DAC2_test, frame.LCOM1_test, frame.LCOM2_test,
frame.LCOM3_test, frame.LCOM4_test, frame.CONNECTIVITY_test, frame.LCOM5_test, frame.COH_test, frame.TCC_test,
frame.LCC_test, frame.ICH_test, frame.WMC_test, frame.NOA_test, frame.NOPA_test, frame.NOP_test, frame.McCABE_test,
frame.BUSWEIMER_test, frame.csm_CDSBP, frame.csm_CC, frame.csm_FD, frame.csm_Blob, frame.csm_SC, frame.csm_MC,
frame.csm_LM, frame.csm_FE, frame.prod_readability, frame.test_readability,frame.Assrtions, frame.Conditions,frame.TryCatch, frame.Loop,frame.Hamcrest,frame.Mockito,
           frame.BadApi,frame.LOC,frame.Expressions, frame.Depth, frame.Vocabulary,
           frame.Understandability,frame.BodySize, frame.Dexterity, frame.NonWhiteCharacters]

    data_x = pd.concat(columns, axis = 1).round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)


def load_all_data_static(frame):
    columns = [frame.no_mutations, frame.isAssertionRoulette, frame.isEagerTest, frame.isLazyTest,
frame.isMysteryGuest, frame.isSensitiveEquality, frame.isResourceOptimism, frame.isForTestersOnly,
frame.isIndirectTesting, frame.LOC_prod, frame.HALSTEAD_prod, frame.RFC_prod, frame.CBO_prod, frame.MPC_prod, frame.IFC_prod, frame.DAC_prod,frame.DAC2_prod, frame.LCOM1_prod, frame.LCOM2_prod,
frame.LCOM3_prod, frame.LCOM4_prod, frame.CONNECTIVITY_prod, frame.LCOM5_prod, frame.COH_prod, frame.TCC_prod,
frame.LCC_prod, frame.ICH_prod, frame.WMC_prod, frame.NOA_prod, frame.NOPA_prod, frame.NOP_prod,
frame.McCABE_prod, frame.BUSWEIMER_prod, frame.LOC_test, frame.HALSTEAD_test, frame.RFC_test, frame.CBO_test,
frame.MPC_test, frame.IFC_test, frame.DAC_test, frame.DAC2_test, frame.LCOM1_test, frame.LCOM2_test,
frame.LCOM3_test, frame.LCOM4_test, frame.CONNECTIVITY_test, frame.LCOM5_test, frame.COH_test, frame.TCC_test,
frame.LCC_test, frame.ICH_test, frame.WMC_test, frame.NOA_test, frame.NOPA_test, frame.NOP_test, frame.McCABE_test,
frame.BUSWEIMER_test, frame.csm_CDSBP, frame.csm_CC, frame.csm_FD, frame.csm_Blob, frame.csm_SC, frame.csm_MC,
frame.csm_LM, frame.csm_FE, frame.prod_readability, frame.test_readability]
    data_x = pd.concat(columns, axis = 1).round(2)
    data_y = pd.concat([frame.mutation], axis = 1)
    return data_x, data_y, len(columns)


def load_all_data_with_mine_static(frame):
    columns = [frame.no_mutations, frame.isAssertionRoulette, frame.isEagerTest, frame.isLazyTest,
frame.isMysteryGuest, frame.isSensitiveEquality, frame.isResourceOptimism, frame.isForTestersOnly,
frame.isIndirectTesting, frame.LOC_prod, frame.HALSTEAD_prod, frame.RFC_prod, frame.CBO_prod, frame.MPC_prod, frame.IFC_prod, frame.DAC_prod,frame.DAC2_prod, frame.LCOM1_prod, frame.LCOM2_prod,
frame.LCOM3_prod, frame.LCOM4_prod, frame.CONNECTIVITY_prod, frame.LCOM5_prod, frame.COH_prod, frame.TCC_prod,
frame.LCC_prod, frame.ICH_prod, frame.WMC_prod, frame.NOA_prod, frame.NOPA_prod, frame.NOP_prod,
frame.McCABE_prod, frame.BUSWEIMER_prod, frame.LOC_test, frame.HALSTEAD_test, frame.RFC_test, frame.CBO_test,
frame.MPC_test, frame.IFC_test, frame.DAC_test, frame.DAC2_test, frame.LCOM1_test, frame.LCOM2_test,
frame.LCOM3_test, frame.LCOM4_test, frame.CONNECTIVITY_test, frame.LCOM5_test, frame.COH_test, frame.TCC_test,
frame.LCC_test, frame.ICH_test, frame.WMC_test, frame.NOA_test, frame.NOPA_test, frame.NOP_test, frame.McCABE_test,
frame.BUSWEIMER_test, frame.csm_CDSBP, frame.csm_CC, frame.csm_FD, frame.csm_Blob, frame.csm_SC, frame.csm_MC,
frame.csm_LM, frame.csm_FE, frame.prod_readability, frame.test_readability,frame.Assrtions, frame.Conditions,frame.TryCatch, frame.Loop,frame.Hamcrest,frame.Mockito,
           frame.BadApi,frame.LOC,frame.Expressions, frame.Depth, frame.Vocabulary,
           frame.Understandability,frame.BodySize, frame.Dexterity, frame.NonWhiteCharacters]

    data_x = pd.concat(columns, axis = 1).round(2)
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

def import_frame(consider_coverage, my_data):
    frame = load_frame()
    frame = load_quartile(frame)
    if consider_coverage and my_data:
        return load_all_data_with_mine(frame)
    if not consider_coverage and my_data:
        return load_all_data_with_mine_static(frame)
    if consider_coverage and not my_data:
        return load_all_data(frame)
    else:
        return load_all_data_static(frame)


# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', activation='linear', init_mode='uniform', dropout_rate=0.1):
    model = keras.Sequential()
    model.add(keras.layers.Dropout(dropout_rate, input_shape=(82,)))
    model.add(keras.layers.Dense(40, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(20, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(2, kernel_initializer=init_mode, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def classification(consider_coverage=True, my_data=True, n_inner=10, n_outer=10):
    """
    Runs the entire process of classification and evaluation
    :param consider_coverage: to include or not the line coverage as a feature
    :param n_inner: number of folds for the inner cross fold validation
    :param n_outer: number of folds for the outer cross fold validation
    :param algorithm: select the algorithm to run; possible choices are 'svc', 'rfc', 'knn' and 'all'
    Validate and save a ML model
    """
    global data_x, data_y, coverage_suffix

    seed = 7
    np.random.seed(seed)

    # the suffix for saving the files
    coverage_suffix = 'dynamic' if consider_coverage else 'static'
    algorithm  = 'my_data' if my_data else ''

    # Import the data
    print('Importing data')

    data_x, data_y, number_of_features = import_frame(consider_coverage, my_data)

    data_x = data_x.values
    data_y = data_y.values

    scaler = StandardScaler()
    scaler.fit(data_x)
    data_x = scaler.transform(data_x)

    print('Import: DONE')

    pipe = Pipeline([('preprocessing', StandardScaler()),
                     ('classifier', KerasClassifier(build_fn=create_model, verbose=0, epochs=2000))])

    # Set up the algorithms to tune, train and evaluate
    #param_grid = get_param_grid(algorithm, metrics)

    # define the grid search parameters
    batch_size = [100] #, 20, 40, 60, 80, 100]
    activation = ['relu'] #['softmax'] #, 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    optimizer = ['Adam'] #['SGD'] #, 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    dropout_rate = [0.2] #, 0.25, 0.3]
    param_grid = dict(batch_size=batch_size, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate)

    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    outer_cv = RepeatedStratifiedKFold(n_splits=n_outer, random_state=seed)

    model = KerasClassifier(build_fn=create_model, verbose=0, epochs=2000)

    early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0003, patience=10, verbose=0, mode='max', restore_best_weights=True)



    # inner cross validation
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=inner_cv,
                        scoring=get_scoring(),
                        #refit = False,
                        refit='roc_auc_scorer',
                        return_train_score=True,
                        verbose=1,
                        n_jobs=-1)


    results = cross_validate(estimator=grid,
                             cv=outer_cv,
                             X=data_x,
                             y=data_y,
                             scoring=get_scoring(),
                             return_train_score=True,
                             verbose=1,
                             n_jobs=40)

    #print(results)

    accuracy = results.get('test_accuracy').mean()
    precision = results.get('test_precision').mean()
    recall = results.get('test_recall').mean()
    f1_score = results.get('test_f1_score').mean()
    roc_auc = results.get('test_roc_auc_scorer').mean()
    mae = results.get('test_mean_absolute_error').mean()
    brier = results.get('test_brier_score').mean()

    print('Performances:\n'
          'Accuracy\t {:.3f}\n'
          'Precision\t {:.3f}\n'
          'Recall\t {:.3f}\n'
          'F1 Score\t {:.3f}\n'
          'ROC AUC\t {:.3f}\n'
          'MAE\t {:.3f}\n'
          'Brier Score\t {:.3f}\n'.format(accuracy, precision, recall, f1_score, roc_auc, mae, brier))

    # save performance metrics
    metrics_res = pd.DataFrame({'accuracy': [accuracy],
                                'precision': [precision],
                                'recall': [recall],
                                'f1_score': [f1_score],
                                'ROC-AUC': [roc_auc],
                                'MAE': [mae],
                                'Brier': [brier]})

    metrics_res.to_csv('{}/evaluation_{}_{}.csv'.format(DATA_DIR, coverage_suffix, algorithm), index=False)

    grid_result = grid.fit(data_x, data_y, callbacks=[early_stopping_monitor])
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print('Best model is:\n{}'.format(grid.best_params_))
    model_string = open('{}/_model_{}_{}.txt'.format(DATA_DIR, coverage_suffix, algorithm), 'w')
    model_string.write(str(model))
    model_string.close()

    print('Saving the model on the entire set')
    #grid.fit(data_x, data_y, callbacks=[early_stopping_monitor])
    #joblib.dump(grid.best_estimator_, '{}/model_{}_{}.pkl'.format(DATA_DIR, coverage_suffix, algorithm), compress=1)


classification(consider_coverage=False)
