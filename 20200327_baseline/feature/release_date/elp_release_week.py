import pandas as pd
from sklearn import preprocessing

def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        print(col)
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


print('read_transformed...')
sell_prices = pd.read_csv('../../../input/sell_prices.csv')
sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"])
print('release_week...')
release_df = sell_prices.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id', 'item_id', 'release']

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'store_id', 'item_id', 'date', 'wm_yr_wk']]


print('merge')
df_feat = pd.merge(df_feat, release_df, on=['store_id', 'item_id'], how='left')
df_feat['elp_week_from_release'] = df_feat['wm_yr_wk'] - df_feat['release']
df_feat['elp_week_from_release'] = df_feat['elp_week_from_release'].apply(lambda x: x if x >= 0 else -99)


df_feat.drop(['store_id', 'item_id', 'wm_yr_wk', 'release'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_elp_week_from_release.pkl')
