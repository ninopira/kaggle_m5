import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]


df_feat['demand_lag_28'] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28))
df_feat['over_zero'] = list(df_feat['demand_lag_28'] == 1)
df_feat['zero'] = list(df_feat['demand_lag_28'] == 0)

for val in [7, 30, 60, 90, 180]:
    colname = f"0demand_lag_28_roll_sum_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["zero"].transform(lambda x: x.rolling(val).sum())
    print(df_feat[colname].mean())

df_feat.drop(['demand', 'demand_lag_28', 'over_zero', 'zero'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_id_zero_demand.pkl')
