import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [28, 35, 60, 120, 393]:
    tr_colname = f"demand_cumsum_lag_{val}"
    print(tr_colname)
    df_feat[tr_colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val).cumsum())

for val in [35, 60, 120, 393]:
    colname = f"demand_cumsum_trend_{val}_28"
    tr_colname = f"demand_cumsum_lag_{val}"
    print(colname)
    df_feat[colname] = (df_feat['demand_cumsum_lag_28'] - df_feat[tr_colname]) / (df_feat[tr_colname] + 1.0)


df_feat.drop(['demand', 'demand_cumsum_lag_28'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_cumsum_demand_trend.pkl')
