import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in range(1, 31):
    colname = f"demand_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
    pkl_name = f'f_id_demand_lag_{val}.pkl'
    print(colname, df_feat[colname].mean())
    print(pkl_name)
    df_feat[['id', 'date', colname]].to_pickle(pkl_name)
    df_feat = df_feat.drop([colname], axis=1)


for base in range(1, 29):
    print('###############################')
    print(base)
    df_feat = df_all[['id', 'demand', 'date']]
    for val in [7, 30, 60, 90, 180, 364]:
        colname = f"demand_lag_{base}_roll_std_{val}"
        print(colname)
        df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(base).rolling(val).std())
        print(colname, df_feat[colname].mean())
    for val in [7, 30, 60, 90, 180, 364]:
        colname = f"demand_lag_{base}_roll_mean_{val}"
        print(colname)
        df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(base).rolling(val).mean())
        print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{base}_roll_skew_30"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(base).rolling(30).skew())
    print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{base}_roll_kurt_30"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(base).rolling(30).kurt())
    print(colname, df_feat[colname].mean())

    df_feat.drop(['demand'], inplace=True, axis=1)
    print(df_feat.head())
    print(df_feat.shape)
    df_feat.to_pickle(f'f_id_lag_demand_{base}_roll.pkl')


df_feat = df_all[['id', 'demand', 'date']]
colname = 'demand_lag_350_rollong_mean_28'
df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(350).rolling(28).mean())
df_feat.drop(['demand'], inplace=True, axis=1)
df_feat.to_pickle(f'f_id_lag_demand_350_roll/28.pkl')