import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [364, 365, 366]:
    colname = f"demand_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
for val in [7, 30, 60, 90, 180]:
    colname = f"demand_lag_364_roll_std_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(364).rolling(val).std())
for val in [7, 30, 60, 90, 180]:
    colname = f"demand_lag_364_roll_mean_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(364).rolling(val).mean())

colname = "demand_lag_364_roll_skew_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(364).rolling(30).skew())

colname = "demand_lag_364_roll_kurt_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(364).rolling(30).kurt())

df_feat.drop(['demand'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_demand_year.pkl')
