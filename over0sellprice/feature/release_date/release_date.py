import pandas as pd
import numpy as np
import os, sys, gc, time, warnings, pickle, psutil, random
from sklearn import preprocessing

print('read_transformed...')
sell_prices = pd.read_csv('../../../input/sell_prices.csv')


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        print(col)
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"])

release_df = sell_prices.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id', 'item_id', 'release']

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'store_id', 'item_id', 'date', 'wm_yr_wk']]

df_merged = pd.merge(df_feat, release_df, on=['store_id', 'item_id'], how='left')
df_merged['elp_week_from_release'] = df_merged['wm_yr_wk'] - df_merged['release']

df_merged.drop(['store_id', 'item_id', 'wm_yr_wk', 'release'], inplace=True, axis=1)

print(df_merged.head())
print(df_merged.tail())
print(df_merged.shape)
df_merged.to_pickle('f_from_release_date.pkl')