"""
28日前前後にそのdeptの中でどれくらいの人気商品なのか
"""

import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date', 'dept_id', 'item_id']]

print('groupby...')
# 日ごとのdeptのdemandのsum
df_date_dept_demand = df_all.groupby(['date', 'dept_id'])['demand'].sum().reset_index()
df_date_dept_demand.columns = ['date', 'dept_id', 'sum_dept_demand']
df_feat = pd.merge(df_feat, df_date_dept_demand, on=['date', 'dept_id'], how='left')

# 日ごとのitemのdemandのsum
df_date_item_demand = df_all.groupby(['date', 'item_id'])['demand'].sum().reset_index()
df_date_item_demand.columns = ['date', 'item_id', 'sum_item_demand']
df_feat = pd.merge(df_feat, df_date_item_demand, on=['date', 'item_id'], how='left')

for val in [28, 29, 30]:
    colname_dept = f"demand_dept_sum_lag_{val}"
    df_feat[colname_dept] = df_feat.groupby(["id"])["sum_dept_demand"].transform(lambda x: x.shift(val))
    colname_item = f"demand_item_sum_lag_{val}"
    df_feat[colname_item] = df_feat.groupby(["id"])["sum_item_demand"].transform(lambda x: x.shift(val))

    colname_portion = f"demand_item_portion_dept_lag_{val}"
    df_feat[colname_portion] = df_feat[colname_item] / df_feat[colname_dept]

for val in [7, 30, 60, 90, 180]:
    colname_dept = f"demand_dept_sum_lag_28_roll_std_{val}"
    print(colname_dept)
    df_feat[colname_dept] = df_feat.groupby(["id"])["sum_dept_demand"].transform(lambda x: x.shift(28).rolling(val).std())

    colname_item = f"demand_item_sum_lag_28_roll_std_{val}"
    print(colname_item)
    df_feat[colname_item] = df_feat.groupby(["id"])["sum_item_demand"].transform(lambda x: x.shift(28).rolling(val).std())


for val in [7, 30, 60, 90, 180]:
    colname_dept = f"demand_dept_sum_lag_28_roll_mean_{val}"
    print(colname_dept)
    df_feat[colname_dept] = df_feat.groupby(["id"])["sum_dept_demand"].transform(lambda x: x.shift(28).rolling(val).mean())

    colname_item = f"demand_item_sum_lag_28_roll_mean_{val}"
    print(colname_item)
    df_feat[colname_item] = df_feat.groupby(["id"])["sum_item_demand"].transform(lambda x: x.shift(28).rolling(val).mean())


colname = "demand_dept_sum_lag_28_roll_skew_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["sum_dept_demand"].transform(lambda x: x.shift(28).rolling(30).skew())

colname = "demand_dept_sum_lag_28_roll_kurt_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["sum_dept_demand"].transform(lambda x: x.shift(28).rolling(30).kurt())

colname = "demand_item_sum_lag_28_roll_skew_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["sum_item_demand"].transform(lambda x: x.shift(28).rolling(30).skew())

colname = "demand_item_sum_lag_28_roll_kurt_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["sum_item_demand"].transform(lambda x: x.shift(28).rolling(30).kurt())

df_feat.drop(['demand', 'dept_id', 'item_id', 'sum_dept_demand', 'sum_item_demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_demand_dept_item.pkl')

