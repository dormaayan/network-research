# -*- coding: utf-8 -*-

import warnings

import numpy as np


import matplotlib
from pca import analyze_componenets
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
 f1_score, roc_auc_score, mean_absolute_error, make_scorer, brier_score_loss

from tensorflow import keras

from data_loader import load_data

warnings.filterwarnings('ignore')

matplotlib.use('Agg')


def get_scoring():
    return dict(accuracy=make_scorer(accuracy_score),
                precision=make_scorer(precision_score),
                recall=make_scorer(recall_score),
                f1_score=make_scorer(f1_score),
                roc_auc_scorer=make_scorer(roc_auc_score),
                mean_absolute_error=make_scorer(mean_absolute_error),
                brier_score=make_scorer(brier_score_loss))

def create_model(optimizer='adam', activation='linear', init_mode='uniform'
                 , dropout_rate=0.1, first_layer=40, second_layer=20, dim = None):
    model = keras.Sequential()
    model.add(keras.layers.Dropout(dropout_rate, input_shape=(dim,)))
    model.add(keras.layers.Dense(first_layer, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(second_layer, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(5, kernel_initializer=init_mode, activation=activation))
    model.add(keras.layers.Dense(2, kernel_initializer=init_mode, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def simpleGrid(top_pca_features = None):
    print('Importing data')

    if top_pca_features == None:
        data_x, data_y, features, number_of_features = load_data(
            effective_non_effective=True, coverage=False,
            grano_test=True, grano_production=True, my_test=True,
            my_production=True, scale=True)
    else:
        data_x, data_y, features, number_of_features = load_data(
            effective_non_effective=True, scale=True, include=analyze_componenets(top_pca_features))
    print('Import: DONE')

    batch_size = [100, 50]
    activation = ['relu']
    optimizer = ['Adam']
    dropout_rate = [0.1, 0.2]
    first_layer = [100, 50]
    second_layer = [20, 10]

    param_grid = dict(
        batch_size=batch_size, optimizer=optimizer, activation=activation,
        dropout_rate=dropout_rate, first_layer=first_layer,
        second_layer=second_layer, dim = [number_of_features])

    inner_cv = StratifiedKFold(n_splits=10, shuffle=True)
    model = KerasClassifier(build_fn=create_model,verbose=0, epochs=2000, batch_size=50)

    early_stopping_monitor = keras.callbacks.EarlyStopping(
        monitor='accuracy', min_delta=0.0003, patience=10,
        verbose=0, mode='max', restore_best_weights=True)


    results = GridSearchCV(
        estimator=model, cv=inner_cv, param_grid=param_grid,
        scoring=get_scoring(), refit='roc_auc_scorer',verbose=20, n_jobs=-1)

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

    print("---------------------------------")
    print('Performances:\n'
          'Accuracy\t {:.3f}\n'
          'Precision\t {:.3f}\n'
          'Recall\t {:.3f}\n'
          'F1 Score\t {:.3f}\n'
          'ROC AUC\t {:.3f}\n'
          'MAE\t {:.3f}\n'
          'Brier Score\t {:.3f}\n'
          .format(accuracy, precision, recall, f1_score, roc_auc, mae, brier))
    print("---------------------------------")

    means = results.cv_results_.get('mean_test_accuracy')
    params = results.cv_results_.get('params')
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))
    return ['{:.3f}'.format(accuracy),
            '{:.3f}'.format(precision),
            '{:.3f}'.format(recall),
            '{:.3f}'.format(f1_score),
            '{:.3f}'.format(roc_auc),
            '{:.3f}'.format(mae),
            '{:.3f}'.format(brier)]


def main():
    simpleGrid(top_pca_features = 36)

if __name__ == '__main__':
    main()
