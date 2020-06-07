# val 週間前との割合
import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)


for base in range(1, 29):
    print(base)
    df_feat = df_all[['id', 'demand', 'date']]
    useless_cols = ['demand']
    for val in [0, 1, 4, 12, 24]:
        attention_day = base + val * 7
        print(attention_day)
        colname = f'demand_lag_{attention_day}_roll_mean_7'
        useless_cols.append(colname)
        df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(attention_day).rolling(7).mean())
        print(colname, df_feat[colname].mean())
        if val > 0:
            new_colname = f'demand_lag_{attention_day}_roll_mean_7_ratio_{base}'
            df_feat[new_colname] = df_feat[colname] / (df_feat[f'demand_lag_{base}_roll_mean_7'] + 0.0001)
            print(new_colname, df_feat[new_colname].mean())

    df_feat.drop(useless_cols, inplace=True, axis=1)
    print(df_feat.shape)
    print(df_feat.head())
    print(df_feat.tail())
    pkl_name = f'f_week_trend_{base}.pkl'
    print(pkl_name)
    df_feat.to_pickle(pkl_name)
