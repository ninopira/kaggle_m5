import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_all = df_all[['id', 'demand', 'date']]

for base in range(1, 29):
    df_feat = df_all.copy()
    df_feat[f'demand_lag_{base}'] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(base))
    df_feat['over_zero'] = list(df_feat[f'demand_lag_{base}'] == 1)
    df_feat['zero'] = list(df_feat[f'demand_lag_{base}'] == 0)

    for val in [7, 30, 60, 90, 180]:
        colname = f"0demand_lag_{base}_roll_sum_{val}"
        df_feat[colname] = df_feat.groupby(["id"])["zero"].transform(lambda x: x.rolling(val).sum())
        print(colname, df_feat[colname].mean())

    df_feat.drop(['demand', f'demand_lag_{base}', 'over_zero', 'zero'], inplace=True, axis=1)

    print(df_feat.head())
    print(df_feat.tail())
    print(df_feat.shape)
    pkl_name = f'f_zero_demand_{base}.pkl'
    print(pkl_name)
    df_feat.to_pickle(pkl_name)
