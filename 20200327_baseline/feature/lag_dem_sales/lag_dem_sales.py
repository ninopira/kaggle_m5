import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date', 'sell_price']]
df_feat['dem_sales'] = df_feat['demand'] * df_feat['sell_price']
print(df_feat['dem_sales'].mean())

for val in [28, 29, 30, 35, 42, 49]:
    colname = f"dem_sales_lag_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["dem_sales"].transform(lambda x: x.shift(val))
for val in [7, 30, 60, 90, 180]:
    colname = f"dem_sales_lag_28_roll_std_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["dem_sales"].transform(lambda x: x.shift(28).rolling(val).std())
for val in [7, 30, 60, 90, 180]:
    colname = f"dem_sales_lag_28_roll_mean_{val}"
    print(colname)
    df_feat[colname] = df_feat.groupby(["id"])["dem_sales"].transform(lambda x: x.shift(28).rolling(val).mean())

colname = "dem_sales_lag_28_roll_skew_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["dem_sales"].transform(lambda x: x.shift(28).rolling(30).skew())

colname = "dem_sales_lag_28_roll_kurt_30"
print(colname)
df_feat[colname] = df_feat.groupby(["id"])["dem_sales"].transform(lambda x: x.shift(28).rolling(30).kurt())

df_feat['4weekdays_mean'] = df_feat[['dem_sales_lag_28', 'dem_sales_lag_35', 'dem_sales_lag_42', 'dem_sales_lag_49']].mean(axis=1)
df_feat['4weekdays_std'] = df_feat[['dem_sales_lag_28', 'dem_sales_lag_35', 'dem_sales_lag_42', 'dem_sales_lag_49']].std(axis=1)
df_feat['4weekdays_min'] = df_feat[['dem_sales_lag_28', 'dem_sales_lag_35', 'dem_sales_lag_42', 'dem_sales_lag_49']].min(axis=1)
df_feat['4weekdays_max'] = df_feat[['dem_sales_lag_28', 'dem_sales_lag_35', 'dem_sales_lag_42', 'dem_sales_lag_49']].max(axis=1)

df_feat.drop(['dem_sales', 'demand', 'sell_price'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_dem_sales.pkl')
