import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [28, 90, 180, 364]:
    colname = f"demand_cumsum_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val).cumsum())

df_feat.drop(['demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_cumsum_demand.pkl')
