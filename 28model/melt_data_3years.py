"""
過去にほぼ同等のことをやっていたがスコアが出なかったので再現
notebookではそれっぽいスコアが出たのでスクリプト化
https://www.kaggle.com/rohitsingh9990/m5-lgbm-fe
3年に伸ばす
"""
import gc
import time
import warnings

from sklearn import preprocessing
import pandas as pd

from reduce_mem import reduce_mem_usage
warnings.filterwarnings('ignore')

nrows = 365 * 3 * 30490   # day * 3year * item

########################
print('########################')
print('read_data')
t0 = time.time()
calendar = pd.read_csv('../input/calendar.csv')
calendar = reduce_mem_usage(calendar)
print('Calendar has {} rows and {} columns'.format(
    calendar.shape[0], calendar.shape[1]))
sell_prices = pd.read_csv('../input/sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)
print('Sell prices has {} rows and {} columns'.format(
    sell_prices.shape[0], sell_prices.shape[1]))
sales_train_validation = pd.read_csv('../input/sales_train_validation.csv')
print('Sales train validation has {} rows and {} columns'.format(
    sales_train_validation.shape[0], sales_train_validation.shape[1]))
submission = pd.read_csv('../input/sample_submission.csv')
t1 = time.time()
print('read_transfomed_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
print('########################')
print('encode_cagorical')
def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        print(col)
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df
t0 = time.time()
print('calender...')
calendar = encode_categorical(calendar,
                              ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)
print('stv...')
sales_train_validation = encode_categorical(sales_train_validation,
                                            ["item_id", "dept_id", "cat_id", "store_id", "state_id"]).pipe(reduce_mem_usage)
print('price..')
sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)
t1 = time.time()
print('encode_cagorical:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
print('########################')
print('melt....')
t0 = time.time()
sales_train_validation = pd.melt(sales_train_validation,
                                 id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                                 var_name='day',
                                 value_name='demand')
sales_train_validation = reduce_mem_usage(sales_train_validation)
print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
t1 = time.time()
print('melt...:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
print('########################')
print('extract_nrows{}....'.format(nrows))
t0 = time.time()
print(sales_train_validation.shape)
sales_train_validation = sales_train_validation.iloc[-nrows:, :]
print(sales_train_validation.shape)
t1 = time.time()
print('extract...:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('seperate_test_dataframes')
t0 = time.time()
test1_rows = [row for row in submission['id'] if 'validation' in row]
test2_rows = [row for row in submission['id'] if 'evaluation' in row]
test1 = submission[submission['id'].isin(test1_rows)]
test2 = submission[submission['id'].isin(test2_rows)]
# change column names
test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931',
                 'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959',
                 'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
t1 = time.time()
print('test_1_shape:{}'.format(test1.shape))
print('test_2_shape:{}'.format(test2.shape))
print('seperate_test_dataframes:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('get product table')
t0 = time.time()
product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
print('product_shape:{}'.format(product.shape))
t1 = time.time()
print('get product table:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('merge with product table for test data')
t0 = time.time()
test2['id'] = test2['id'].str.replace('_evaluation', '_validation')
test1 = test1.merge(product, how='left', on='id')
test2 = test2.merge(product, how='left', on='id')
test2['id'] = test2['id'].str.replace('_validation', '_evaluation')
print('test_1_shape:{}'.format(test1.shape))
print('test_2_shape:{}'.format(test2.shape))
t1 = time.time()
print('merge with product table for test data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('melt_test_data')
print('before_test_1_shape:{}'.format(test1.shape))
print('before_test_2_shape:{}'.format(test2.shape))
t0 = time.time()
test1 = pd.melt(test1,
                id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                var_name='day',
                value_name='demand')
test2 = pd.melt(test2,
                id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                var_name='day', value_name='demand')
print('after_test_1_shape:{}'.format(test1.shape))
print('after_test_2_shape:{}'.format(test2.shape))
t1 = time.time()
print('melt_test_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
print('########################')
print('cancat_train_test')
sales_train_validation['part'] = 'train'
test1['part'] = 'test1'
test2['part'] = 'test2'

print('train:{}'.format(sales_train_validation.shape))
print('test1:{}'.format(test1.shape))
print('test2:{}'.format(test2.shape))
t0 = time.time()
data = pd.concat([sales_train_validation, test1, test2], axis=0)
del sales_train_validation, test1, test2
gc.collect()
print('cancat_data_shape:{}'.format(data.shape))
t1 = time.time()
print('melt_test_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
print('########################')
print('merge')

# drop some calendar features
calendar.drop(['weekday', 'wday', 'month', 'year'], inplace=True, axis=1)
# notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
print('calende_merge')
t0 = time.time()
data = pd.merge(data, calendar, how='left', left_on=['day'], right_on=['d'])
data.drop(['d', 'day'], inplace=True, axis=1)
print(data.shape)
t1 = time.time()
print('calender_merge:{0}'.format(t1-t0) + '[sec]')

print('sales_merge')
t0 = time.time()
data = data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
print(data.shape)
t1 = time.time()
print('sales_merge:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

len_data = len(data)
print('writing..')
t0 = time.time()
data.to_pickle('{}_{}_melt.pkl'.format(len_data, nrows))
t1 = time.time()
print('writing:{0}'.format(t1-t0) + '[sec]')

