import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date', 'store_id']]

df_tmp = df_feat.groupby(['date', 'store_id'])['demand'].sum().reset_index()
df_tmp.columns = ['date', 'store_id', 'demand_store_sum']

for val in [28, 90, 180, 364]:
    colname = f"demand_store_sum_cumsum_lag_{val}"
    print(colname)
    df_tmp[colname] = df_tmp.groupby(['store_id'])["demand_store_sum"].transform(lambda x: x.shift(val).cumsum())
    print(df_tmp[colname].mean())

df_tmp.drop(['demand_store_sum'], inplace=True, axis=1)
print(df_tmp.shape)
print(df_tmp.head())

df_feat = pd.merge(df_feat, df_tmp, on=['date', 'store_id'], how='left')
df_feat.drop(['store_id', 'demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_cumsum_shop_demand.pkl')
