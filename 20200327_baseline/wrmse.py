import gc

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import pandas as pd
from reduce_mem import reduce_mem_usage


def _prepare_prod():
    print('preparea_prod...')
    sales_train_validation = pd.read_csv('../input/sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    print('encode_cagorical')

    def encode_categorical(df, cols):
        for col in cols:
            # Leave NaN as it is.
            le = preprocessing.LabelEncoder()
            not_null = df[col][df[col].notnull()]
            df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
        return df

    print('stv...')
    sales_train_validation = encode_categorical(sales_train_validation, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]).pipe(reduce_mem_usage)
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    print('done')
    return product


def _create_weigft_mat(product, NUM_ITEMS=30490):
    print('preparea_weight_mat...')
    weight_mat = np.c_[
        np.ones([NUM_ITEMS, 1]).astype(np.int8),  # level 1
        pd.get_dummies(product.state_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.store_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.cat_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.dept_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.item_id.astype(str), drop_first=False).astype('int8').values,
        pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str), drop_first=False).astype('int8').values,
        np.identity(NUM_ITEMS).astype(np.int8)  # item :level 12
        ].T

    weight_mat_csr = csr_matrix(weight_mat)
    print('done')
    return weight_mat_csr


def weight_calc(data):
    print('weight_calc...')
    product = _prepare_prod()
    weight_mat_csr = _create_weigft_mat(product)
    # calculate the denominator of RMSSE, and calculate the weight base on sales amount

    sales_train_val = pd.read_csv('../input/sales_train_validation.csv')

    d_name = ['d_' + str(i+1) for i in range(1913)]

    sales_train_val = weight_mat_csr * sales_train_val[d_name].values

    # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
    df_tmp = ((sales_train_val > 0) *
              np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1)))

    start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1)-1

    flag = np.dot(np.diag(1/(start_no+1)),
                  np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1))) < 1

    sales_train_val = np.where(flag, np.nan, sales_train_val)

    # denominator of RMSSE / RMSSEの分母
    weight1 = np.nansum(np.diff(sales_train_val, axis=1)
                        ** 2, axis=1)/(1913-start_no)

    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') &
                  (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp = df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp[product.id].values

    weight2 = weight_mat_csr * df_tmp

    weight2 = weight2/np.sum(weight2)

    del sales_train_val
    gc.collect()
    print('done')
    return weight1, weight2, weight_mat_csr
