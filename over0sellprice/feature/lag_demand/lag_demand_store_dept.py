import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date', 'store_id', 'dept_id']]

print('groupby')
df_tmp = df_feat.groupby(['date', 'store_id', 'dept_id'])['demand'].sum().reset_index()
df_tmp.columns = ['date', 'store_id', 'dept_id', 'demand_store_dept_sum']
print(df_tmp.shape)

for val in [28, 29, 30]:
    colname = f"demand_store_dept_lag_{val}"
    print(colname)
    df_tmp[colname] = df_tmp.groupby(['store_id', 'dept_id'])["demand_store_dept_sum"].transform(lambda x: x.shift(val))
    print(df_tmp[colname].mean())

for val in [7, 30, 60, 90, 180]:
    colname = f"demand_store_dept_lag_28_roll_std_{val}"
    print(colname)
    df_tmp[colname] = df_tmp.groupby(['store_id', 'dept_id'])["demand_store_dept_sum"].transform(lambda x: x.shift(28).rolling(val).std())
    print(df_tmp[colname].mean())

for val in [7, 30, 60, 90, 180]:
    colname = f"demand_store_dept_lag_28_roll_mean_{val}"
    print(colname)
    df_tmp[colname] = df_tmp.groupby(['store_id', 'dept_id'])["demand_store_dept_sum"].transform(lambda x: x.shift(28).rolling(val).mean())
    print(df_tmp[colname].mean())

colname = "demand_store_dept_lag_28_roll_skew_30"
print(colname)
df_tmp[colname] = df_tmp.groupby(['store_id', 'dept_id'])["demand_store_dept_sum"].transform(lambda x: x.shift(28).rolling(30).skew())
print(df_tmp[colname].mean())

colname = "demand_store_dept_lag_28_roll_kurt_30"
print(colname)
df_tmp[colname] = df_tmp.groupby(['store_id', 'dept_id'])["demand_store_dept_sum"].transform(lambda x: x.shift(28).rolling(30).kurt())
print(df_tmp[colname].mean())

df_tmp.drop(['demand_store_dept_sum'], inplace=True, axis=1)


df_feat = pd.merge(df_feat, df_tmp, on=['date', 'store_id', 'dept_id'], how='left')
df_feat.drop(['store_id', 'dept_id', 'demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_store_dept_lag_demand.pkl')
