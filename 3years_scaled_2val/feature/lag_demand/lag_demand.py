import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [28, 29, 30]:
    colname = f"demand_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
for val in [7, 30, 60, 90, 180]:
    colname = f"demand_lag_28_roll_std_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).std())
for val in [7, 30, 60, 90, 180]:
    colname = f"demand_lag_28_roll_mean_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).mean())

colname = "demand_lag_28_roll_skew_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).skew())

colname = "demand_lag_28_roll_kurt_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).kurt())

df_feat.drop(['demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_demand.pkl')
