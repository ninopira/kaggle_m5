import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in range(29, 31):
    colname = f"demand_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
    pkl_name = f'f_id_demand_lag_{val}.pkl'
    print(colname, df_feat[colname].mean())
    print(pkl_name)
    df_feat[['id', 'date', colname]].to_pickle(pkl_name)
    df_feat = df_feat.drop([colname], axis=1)
