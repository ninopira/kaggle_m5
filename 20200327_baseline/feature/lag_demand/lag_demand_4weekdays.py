import pandas as pd
import numpy as np
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [28, 35, 42, 49]:
    colname = f"demand_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val))

df_feat['4weekdays_mean'] = df_feat[['demand_lag_28', 'demand_lag_35', 'demand_lag_42', 'demand_lag_49']].mean(axis=1)
df_feat['4weekdays_std'] = df_feat[['demand_lag_28', 'demand_lag_35', 'demand_lag_42', 'demand_lag_49']].std(axis=1)
df_feat['4weekdays_min'] = df_feat[['demand_lag_28', 'demand_lag_35', 'demand_lag_42', 'demand_lag_49']].min(axis=1)
df_feat['4weekdays_max'] = df_feat[['demand_lag_28', 'demand_lag_35', 'demand_lag_42', 'demand_lag_49']].max(axis=1)


df_feat.drop(['demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_demand_4weekdays_stat.pkl')
