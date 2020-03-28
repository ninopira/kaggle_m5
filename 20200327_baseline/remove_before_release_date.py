"""
item_idが登場する前のデータは不要
https://www.kaggle.com/kyakovlev/m5-simple-fe
"""
import pandas as pd
import numpy as np
import os, sys, gc, time, warnings, pickle, psutil, random
from reduce_mem import reduce_mem_usage
from sklearn import preprocessing


# Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    # import ipdb; ipdb.set_trace()
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        print(col)
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


sell_prices = pd.read_csv('../input/sell_prices.csv')
sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"])

# 発売日を取得
release_df = sell_prices.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id', 'item_id', 'release']
release_df = reduce_mem_usage(release_df)
# 元のdmに連結
grid_df = pd.read_pickle('./23965140_22257700_melt.pkl')
print('grid_df_shape:{}'.format(grid_df.shape))

grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
print('merge_release_date_grid_df_shape:{}'.format(grid_df.shape))

# calendar = pd.read_csv('../input/calendar.csv')
# grid_df = merge_by_concat(grid_df, calendar[['wm_yr_wk', 'date']], ['date'])
# print('merge_calender_grid_df_shape:{}'.format(grid_df.shape))

# 発売日前のデータを削除
grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']]
grid_df = grid_df.reset_index(drop=True)
print('remove_before_release_date_grid_df_shape:{}'.format(grid_df.shape))

# other columns in case we will need it
grid_df['release'] = grid_df['release'] - grid_df['release'].min()

print('writing..')
grid_df.to_pickle('{}_removed_before_release_day_based_don_23965140_22257700_melt.pkl'.format(len(grid_df)))
