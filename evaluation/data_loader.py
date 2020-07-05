# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler


CSV_PATH = "../complete-frame.csv"
CSV_MINER_PATH = "../testminereffectiveness-extended.csv"
DATA_DIR = "results"


line_coverage = ['line_coverage']

grano_size = [
    'LOC',
    'NOA',
    'NOPA',
    'NOP']

grano_code_smells = [
    'csm_CDSBP',
    'csm_CC',
    'csm_FD',
    'csm_Blob',
    'csm_SC',
    'csm_MC',
    'csm_LM',
    'csm_FE']

grano_test_smells = [
    'isAssertionRoulette',
    'isEagerTest',
    'isLazyTest',
    'isMysteryGuest',
    'isSensitiveEquality',
    'isResourceOptimism',
    'isForTestersOnly',
    'isIndirectTesting']

grano_literature = [
    'HALSTEAD',
    'RFC',
    'CBO',
    'LCOM1',
    'LCOM2',
    'LCOM3',
    'LCOM4',
    'LCOM5',
    'WMC',
    'McCABE',
    'BUSWEIMER']

grano_complexity = [
    'MPC',
    'IFC',
    'DAC',
    'DAC2',
    'CONNECTIVITY',
    'COH',
    'TCC',
    'LCC',
    'ICH']

my_textual = [
    'Word',
    'Special',
    'Non Whithe Characters',
    'Comments',
    'No. Methods',
    'Vocabulary',
    'Strings']


my_ast_shape = [
    'AST size',
    'Deg^2',
    'Deg^3',
    'Deg',
    'Deg^-1',
    'Deg^-2',
    'Max Depth',
    'Avg Depth^(-2)',
    'Avg Depth^(-1)',
    'Avg Depth',
    'Avg Depth^2',
    'Avg Depth^3',
    'Decendent',
    'DegPerm']


my_ast_types = [
    'No. Method Invoctions',
    'Dexterity',
    'No. Expressions',
    'Numeric Literals',
    'No. Field Access',
    'No. Primitives']


my_mccabe_style = [
    'No. &&',
    'No. ||',
    'No. Try',
    'No. Catch',
    'No. Loop',
    'No. Break',
    'No. Continue',
    'No. Conditions',
    'No. Else',
    'No. Ternary'
]

my_test_api = [
    'Bad API',
    'Junit',
    'Hamcrest',
    'Mockito']

my_general = my_textual + my_ast_shape + my_ast_types + my_mccabe_style

my_textual_test = my_textual
my_ast_shape_test = my_ast_shape
my_ast_types_test = my_ast_types
my_mccabe_style_test = my_mccabe_style

my_textual_production = [(factor + "_production") for factor in my_textual]
my_ast_shape_production = [(factor + "_production") for factor in my_ast_shape]
my_ast_types_production = [(factor + "_production") for factor in my_ast_types]
my_mccabe_style_production = [(factor + "_production") for factor in my_mccabe_style]


my_general_test = my_textual_test + my_ast_shape_test + my_ast_types_test + my_mccabe_style_test
my_general_production = my_textual_production + my_ast_shape_production + my_ast_types_production + my_mccabe_style_production


grano_size_test = [(factor + "_test") for factor in grano_size]
grano_literature_test = [(factor + "_test") for factor in grano_literature] + ['test_readability']
grano_complexity_test = [(factor + "_test") for factor in grano_complexity]


grano_size_production = [(factor + "_prod") for factor in grano_size]
grano_literature_production = [(factor + "_prod") for factor in grano_literature] + ['prod_readability']
grano_complexity_production = [(factor + "_prod") for factor in grano_complexity]


grano_general_production = grano_size_production + grano_literature_production + grano_complexity_production
grano_general_test = grano_size_test + grano_literature_test + grano_complexity_test

grano_production_data = grano_general_production + grano_code_smells
grano_test_data = grano_general_test + grano_test_smells


my_test_data = my_general_test + my_test_api
my_production_data = my_general_production


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
    return frame


def load_frame():

    d = dict(zip(my_general,[(factor + "_production") for factor in my_general]))
    d['TestClassName'] = 'ClassName'

    frame1 = pd.read_csv(CSV_PATH, sep=",")
    frame1 = frame1.sample(frac=1).reset_index(drop=True)
    frame1['TestClassName'] = frame1.apply(lambda row: label_rename1(row), axis=1)
    frame1['ClassName'] = frame1.apply(lambda row: label_rename2(row), axis=1)

    frame2 = pd.read_csv(CSV_MINER_PATH, sep=',')

    frame3 = pd.read_csv(CSV_MINER_PATH, sep=',')
    frame3 = frame3.rename(columns=d)
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
    return [x for x in lst if x not in values_as_set]


def strip(feature):
    return feature.split('_')[0]

def get_category(feature):
    if feature in line_coverage:
        return 'grano', 'test', 'line-coverage'
    if feature in my_textual_test:
        return 'mine', 'test', 'textual'
    if feature in my_ast_shape_test:
        return 'mine', 'test', 'ast-shape'
    if feature in my_ast_types_test:
        return 'mine', 'test', 'ast-types'
    if feature in my_mccabe_style_test:
        return 'mine', 'test', 'mccabe-style'
    if feature in my_test_api:
        return 'mine', 'test', 'api'

    if feature in my_textual_production:
        return 'mine', 'production', 'textual'
    if feature in my_ast_shape_production:
        return 'mine', 'production', 'ast-shape'
    if feature in my_ast_types_production:
        return 'mine', 'production', 'ast-types'
    if feature in my_mccabe_style_production:
        return 'mine', 'production', 'mccabe-style'

    if feature in grano_size_test:
        return 'grano', 'test', 'size'
    if feature in grano_literature_test:
        return 'grano', 'test', 'literature'
    if feature in grano_complexity_test:
        return 'grano', 'test', 'complexity'
    if feature in grano_test_smells:
        return 'grano', 'test', 'test-smell'

    if feature in grano_size_production:
        return 'grano', 'production', 'size'
    if feature in grano_literature_production:
        return 'grano', 'production', 'literature'
    if feature in grano_complexity_production:
        return 'grano', 'production', 'complexity'
    if feature in grano_code_smells:
        return 'grano', 'production', 'code-smell'


def pick_data(coverage, grano_test, grano_production, my_test, my_production):
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
        res += my_production_data
    return res


def load_data(effective_non_effective = False,coverage = False, grano_test = False,
              grano_production = False, my_test = False, my_production = False,
              scale = True, include = []):
    frame = load_frame()
    if effective_non_effective:
        frame = load_quartile(frame)
    columns = []
    if include == []:
        columns = pick_data(coverage, grano_test, grano_production, my_test, my_production)
    else:
        columns = include

    data_x = frame[columns]
    data_y = pd.concat([frame.mutation], axis = 1)

    if scale:
        scaler = StandardScaler()
        scaler.fit(data_x)
        data_x = scaler.transform(data_x)

    return data_x, data_y, columns, len(columns)

def main():
    x,y,c,l = load_data(
        coverage = True, grano_test = True, grano_production = True,
        my_test = True, my_production = True)
    print(l)


if __name__ == '__main__':
    main()
