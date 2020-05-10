import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in range(1, 29):
    colname = f"demand_cumsum_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val).cumsum())
    print(colname, df_feat[colname].mean())
    print(df_feat.head())
    print(df_feat.tail())
    pkl_name = f'f_id_cumsum_demand_{val}.pkl'
    print(pkl_name)
    df_feat[['id', 'date', colname]].to_pickle(pkl_name)
    df_feat = df_feat.drop([colname], axis=1)


for val in [90, 120, 180, 220, 364]:
    colname = f"demand_cumsum_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val).cumsum())
    print(colname, df_feat[colname].mean())
    print(df_feat.head())
    print(df_feat.tail())
    pkl_name = f'f_id_cumsum_demand_{val}.pkl'
    print(pkl_name)
    df_feat[['id', 'date', colname]].to_pickle(pkl_name)
    df_feat = df_feat.drop([colname], axis=1)
