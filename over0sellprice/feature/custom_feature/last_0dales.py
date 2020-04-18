import numpy as np
import pandas as pd
import os, sys, gc, warnings, psutil, random
import pandas as pd
import sys
sys.path.append('../../')
warnings.filterwarnings('ignore')

print('read_transformed...')
grid_df = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(grid_df.shape)

TARGET = 'demand'

print('nonzero..')
n_day = 1
ls_df = grid_df[['id', 'date', TARGET]]
ls_df['non_zero'] = (ls_df[TARGET] > 0).astype(np.int8)
ls_df['non_zero_lag'] = ls_df.groupby(['id'])['non_zero'].transform(lambda x: x.shift(n_day).rolling(2000, 1).sum()).fillna(-1)


temp_df = ls_df[['id', 'date', 'non_zero_lag']].drop_duplicates(subset=['id', 'non_zero_lag'])
temp_df.columns = ['id', 'd_min', 'non_zero_lag']
print('merge..')
ls_df = ls_df.merge(temp_df, on=['id', 'non_zero_lag'], how='left')

print('diff')
ls_df['last_sale'] = (pd.to_datetime(ls_df['date']) - pd.to_datetime(ls_df['d_min']))
print('days')
ls_df['last_sale_days'] = ls_df['last_sale'].apply(lambda x: x.days)

grid_df = ls_df[['id', 'date', 'last_sale_days']]

print(grid_df.head())
print(grid_df.tail())
print(grid_df.shape)

grid_df.to_pickle('f_last_sell.pkl')
