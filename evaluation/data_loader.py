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

grano_general = grano_size + grano_literature + grano_complexity

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

grano_general_production = [(factor + "_prod") for factor in grano_general]
grano_general_test = [(factor + "_test") for factor in grano_general]

grano_production_data = grano_general_production + grano_code_smells + ['prod_readability']
grano_test_data = grano_general_test + grano_test_smells + ['test_readability']

my_general_test = my_general
my_general_production = [(factor + "_production") for factor in my_general]

my_test_data = my_general + my_test_api
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


def get_category(feature):
    if feature in line_coverage:
        return 'line_coverage'
    if feature  in grano_test_data:
        return 'grano_test'
    if feature in grano_production_data:
        return 'grano_production'
    if feature in my_test_data:
        return 'my_test'
    if feature in my_production_data:
        return 'my_production'

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
