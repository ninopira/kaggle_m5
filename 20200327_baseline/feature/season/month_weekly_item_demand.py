import gc
import pandas as pd
import sys
sys.path.append('../../')
from reduce_mem import reduce_mem_usage
import numpy as np

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt.pkl')
print(df_all.shape)

df_all['tmp'] = pd.to_datetime(df_all['date'])
df_all['dayofweek'] = getattr(df_all['tmp'].dt, 'dayofweek').astype(np.int8)
df_all['month'] = getattr(df_all['tmp'].dt, 'month').astype(np.int8)

# demandが0以上に限定
df_tmp = df_all.query('demand > 0')
print('>0')
print(df_tmp.shape)


df_iteim_dayofweek_month_demand_mean = df_tmp.groupby(['item_id', 'dayofweek', 'month'])['demand'].mean().reset_index()
df_iteim_dayofweek_month_demand_mean.columns = ['item_id', 'dayofweek', 'month', 'month_weekly_item_demand']

df_all = df_all[['id', 'date', 'item_id', 'dayofweek', 'month']]
df_f = pd.merge(df_all, df_iteim_dayofweek_month_demand_mean,
                on=['item_id', 'dayofweek', 'month'],
                how='left')

df_f = df_f[['id', 'date', 'month_weekly_item_demand']]
print(df_f.head())
print(df_f.shape)


df_f.to_pickle('f_month_dayofweek_item_demand.pkl')
