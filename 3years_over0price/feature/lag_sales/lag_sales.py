import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'sell_price', 'date']]

print('lag_price_t1')
df_feat['lag_price_t1'] = df_feat.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
print('price_change_t1')
df_feat['price_change_t1'] = (df_feat['lag_price_t1'] - df_feat['sell_price']) / (df_feat['lag_price_t1'])
print('price_rolling_365_max')
df_feat['price_rolling_365_max'] = df_feat.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
df_feat['price_change_t365'] = (df_feat['price_rolling_365_max'] - df_feat['sell_price']) / (df_feat['price_rolling_365_max'])
print('price_rolling_std_t7')
df_feat['price_rolling_std_t7'] = df_feat.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
print('price_rolling_std_t30')
df_feat['price_rolling_std_t30'] = df_feat.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())

df_feat.drop(['price_rolling_365_max', 'lag_price_t1', 'sell_price'], inplace=True, axis=1)


print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_id_lag_sales.pkl')
