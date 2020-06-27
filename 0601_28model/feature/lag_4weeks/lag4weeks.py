
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_all = df_all[['id', 'demand', 'date']]


for val in range(1, 29):
    df_feat = df_all.copy()
    weeks = [val, val+7, val+14, val+21]
    cols = []
    for day in weeks:
        colname = f"demand_lag_{day}"
        print(colname)
        df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(day))
        cols.append(colname)

    print('sumarry')

    df_feat['4weekdays_mean'] = df_feat[cols].mean(axis=1)
    df_feat['4weekdays_std'] = df_feat[cols].std(axis=1)
    df_feat['4weekdays_min'] = df_feat[cols].min(axis=1)
    df_feat['4weekdays_max'] = df_feat[cols].max(axis=1)

    df_feat = df_feat.drop(['demand', f"demand_lag_{val}"], axis=1)
    print(df_feat['4weekdays_mean'].mean())

    print(df_feat.head())
    print(df_feat.tail())
    print(df_feat.shape)
    print(f'f_id_lag_demand_4weekdays_stat_{val}.pkl')
    df_feat.to_pickle(f'f_id_lag_demand_4weekdays_stat_{val}.pkl')
