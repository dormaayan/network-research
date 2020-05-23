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
    frame3 = frame3.rename(columns = d, errors = 'raise')
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

def get_scoring():
    """Returns the scores to evaluate the model"""
    return dict(accuracy=make_scorer(accuracy_score),
                precision=make_scorer(precision_score),
                recall=make_scorer(recall_score),
                f1_score=make_scorer(f1_score),
                roc_auc_scorer=make_scorer(roc_auc_score),
                mean_absolute_error=make_scorer(mean_absolute_error),
                brier_score=make_scorer(brier_score_loss))

def import_frame(consider_coverage):
    frame = load_frame()
    frame = load_quartile(frame)
    return load_all_their_data(frame)
    #if consider_coverage:
    #    return load_all_data_dynamic(frame)
    #return load_all_data(frame)


# Function to create model, required for KerasClassifier
"""
def create_model(optimizer='adam', activation='linear', init_mode='uniform'
, dropout_rate=0.1, first_layer=40, second_layer=20):
    model = keras.Sequential()
    model.add(keras.layers.Dropout(dropout_rate, input_shape=(84,)))
    model.add(keras.layers.Dense(first_layer, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(second_layer, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(5, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(2, kernel_initializer=init_mode, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
"""



def create_model2( nl1=1, nl2=1,  nl3=1,
nn1=1000, nn2=500, nn3 = 200, lr=0.01, decay=0., l1=0.01, l2=0.01,
act = 'relu', dropout=0, optimizer='Adam', input_shape=66, output_shape=2):

    #opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,  decay=decay)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

    model = keras.Sequential()

    # for the firt layer we need to specify the input dimensions
    first=True

    for i in range(nl1):
        if first:
            model.add(keras.layers.Dense(nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        else:
            model.add(keras.layers.Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout!=0:
            model.add(keras.layers.Dropout(dropout))

    for i in range(nl2):
        if first:
            model.add(keras.layers.Dense(nn2, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        else:
            model.add(keras.layers.Dense(nn2, activation=act, kernel_regularizer=reg))
        if dropout!=0:
            model.add(keras.layers.Dropout(dropout))

    for i in range(nl3):
        if first:
            model.add(keras.layers.Dense(nn3, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        else:
            model.add(keras.layers.Dense(nn3, activation=act, kernel_regularizer=reg))
        if dropout!=0:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(output_shape, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer = optimizer,
     metrics=['accuracy'])
     #optimizer=opt, metrics=['accuracy'])
    return model

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', activation='linear', init_mode='uniform'
, dropout_rate=0.1, first_layer=40, second_layer=20):
    model = keras.Sequential()
    model.add(keras.layers.Dropout(dropout_rate, input_shape=(66,)))
    model.add(keras.layers.Dense(first_layer, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(second_layer, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(5, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(2, kernel_initializer=init_mode, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def simpleGrid(consider_coverage, n_inner=10):
    """
    Runs the entire process of classification and evaluation
    :param consider_coverage: to include or not the line coverage as a feature
    :param n_inner: number of folds for the inner cross fold validation
    :param n_outer: number of folds for the outer cross fold validation
    :param algorithm: select the algorithm to run; possible choices are 'svc', 'rfc', 'knn' and 'all'
    Validate and save a ML model
    """
    global data_x, data_y, coverage_suffix

    #seed = 7
    #np.random.seed(seed)

    # the suffix for saving the files
    coverage_suffix = 'dynamic' if consider_coverage else 'static'
    algorithm  = '' #'my_data' if my_data else ''

    # Import the data
    print('Importing data')

    data_x, data_y, number_of_features = import_frame(consider_coverage)

    data_x = data_x.values
    data_y = data_y.values

    scaler = StandardScaler()
    scaler.fit(data_x)
    data_x = scaler.transform(data_x)

    print('Import: DONE')

    # learning algorithm parameters
    #lr=[1e-2, 1e-3, 1e-4]
    #decay=[1e-6,1e-9,0]

    # activation
    #activation= ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] #['relu'] #, 'sigmoid']

    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

    # numbers of layers
    #nl1 = [0,1,2,3]
    #nl2 = [0,1,2,3]
    #nl3 = [0,1,2,3]

    # neurons in each layer
    #nn1=[300,700,1400, 2100]
    #nn2=[100,400,800]
    #nn3=[50,150,300]

    # dropout and regularisation
    #dropout = [0, 0.1, 0.2, 0.3]
    #l1 = [0, 0.01, 0.003, 0.001,0.0001]
    #l2 = [0, 0.01, 0.003, 0.001,0.0001]


    # dictionary summary
    #param_grid = dict(
    #                    nl1=nl1, nl2=nl2, nl3=nl3, nn1=nn1, nn2=nn2, nn3=nn3,
    #                    act=activation, l1=l1, l2=l2, dropout=dropout, optimizer=optimizer)
    #                    # lr=lr, decay=decay, dropout=dropout)

        # define the grid search parameters

    batch_size = [100,50] #[10, 20, 40, 60, 80, 100]
    activation = ['relu',] #['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    optimizer = ['Adam','Adamax'] #['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    dropout_rate = [0.1,0.2] #[0.0 ,0.1 ,0.2, 0.25, 0.3]
    first_layer = [1000, 100,500] #, 80, 70, 60, 50, 40] #, 30, 20, 10]
    second_layer = [20,10,5] #[50, 40, 30, 20, 10]
    param_grid = dict(batch_size=batch_size, optimizer=optimizer,
     activation=activation, dropout_rate=dropout_rate,
     first_layer=first_layer, second_layer=second_layer)

    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True)

    model = KerasClassifier(build_fn=create_model,
     verbose=0, epochs=2000, batch_size=50)

    early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0003, patience=10, verbose=0, mode='max', restore_best_weights=True)



    # inner cross validation

    results = RandomizedSearchCV(estimator=model, cv=inner_cv,
    param_distributions=param_grid, scoring=get_scoring(), refit='roc_auc_scorer',
    verbose=20, n_iter=10, n_jobs=-1)

    results.fit(data_x, data_y, callbacks=[early_stopping_monitor])

    print("-----------------------------")
    print(results.cv_results_.get('mean_test_accuracy'))
    print(max(results.cv_results_.get('mean_test_accuracy')))
    #values.index(max(values))
    print('The best configuration is {}'.format(results.best_params_))
    config_index = np.argmax(results.cv_results_.get('mean_test_accuracy'))
    print(config_index)
    print("-----------------------------")
    accuracy = results.cv_results_.get('mean_test_accuracy')[config_index]
    precision = results.cv_results_.get('mean_test_precision')[config_index] #.mean()
    recall = results.cv_results_.get('mean_test_recall')[config_index] #.mean()
    f1_score = results.cv_results_.get('mean_test_f1_score')[config_index] #.mean()
    roc_auc = results.cv_results_.get('mean_test_roc_auc_scorer')[config_index] #.mean()
    mae = results.cv_results_.get('mean_test_mean_absolute_error')[config_index] #.mean()
    brier = results.cv_results_.get('mean_test_brier_score')[config_index] #.mean()


    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print("---------------------------------")
    print('Performances:\n'
          'Accuracy\t {:.3f}\n'
          'Precision\t {:.3f}\n'
          'Recall\t {:.3f}\n'
          'F1 Score\t {:.3f}\n'
          'ROC AUC\t {:.3f}\n'
          'MAE\t {:.3f}\n'
          'Brier Score\t {:.3f}\n'.format(accuracy, precision, recall, f1_score, roc_auc, mae, brier))
    print("---------------------------------")

    means = results.cv_results_.get('mean_test_accuracy')
    #stds = results.cv_results_.get('std_mean_test_accuracy')
    params = results.cv_results_.get('params')
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))

    # save performance metrics

    """

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

    """


simpleGrid(consider_coverage=False)
