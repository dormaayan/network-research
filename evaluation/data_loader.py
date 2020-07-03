import numpy as np
import pandas as pd



from sklearn.preprocessing import StandardScaler


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
              'Avg Depth^3',
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


grano_production_data = [(factor + "_prod") for factor in grano_general] + code_smells + ['prod_readability']
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

def delete_by_values(lst, values):
    values_as_set = set(values)
    return [x for x in lst if x not in values_as_set]

def pick_data(coverage, grano_test, grano_production, my_test, my_production, exclude):
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
    return delete_by_values(res, exclude)

def load_data(effective_non_effective = False,coverage = False, grano_test = False,
              grano_production = False, my_test = False, my_production = False,
              scale = True, exclude = []):
    frame = load_frame()
    if effective_non_effective:
        frame = load_quartile(frame)
    columns = pick_data(coverage, grano_test, grano_production, my_test, my_production, exclude)
    data_x = frame[columns]
    data_y = pd.concat([frame.mutation], axis = 1)

    if scale:
        scaler = StandardScaler()
        scaler.fit(data_x)
        data_x = scaler.transform(data_x)

    return data_x, data_y, columns, len(columns)

def main():
    x,y,c,l = load_data(coverage = True,
                         grano_test = True, grano_production = True, my_test = True, my_production = True)
    print(l)


if __name__ == '__main__':
    main()
