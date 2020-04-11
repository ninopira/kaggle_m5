import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [7, 30, 60, 90, 180]:
    colname = f"demand_lag_28_ewm_mean_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28).ewm(span=val).mean())
    print(df_feat[colname].mean())


df_feat.drop(['demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_demand_ewm.pkl')
