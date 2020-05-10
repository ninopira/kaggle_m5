# 一部、lag_demandと重複しているが10列くらいだし気にしない
import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]

for val in [0, 1, 4, 12, 24]:
    attention_day = 28 + val * 7
    print(attention_day)
    # こいつらいらないかも。
    colname = f'demand_lag_{attention_day}_roll_mean_7'
    df_feat[colname] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(attention_day).rolling(7).mean())
    print(colname, df_feat[colname].mean())
    if val > 0:
        new_colname = f'demand_lag_{attention_day}_roll_mean_7_ratio_28'
        df_feat[new_colname] = df_feat[colname] / (df_feat['demand_lag_28_roll_mean_7'] + 0.0001)
        print(new_colname, df_feat[new_colname].mean())


df_feat.drop(['demand', 'demand_lag_28_roll_mean_7'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_week_trend.pkl')
