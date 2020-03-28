import pandas as pd
import sys
from sklearn import preprocessing
sys.path.append('../../')


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        print(col)
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


df_all = pd.read_pickle('../../23443584_removed_before_release_day_based_don_23965140_22257700_melt.pkl')
df_all = df_all[['id', 'date', 'store_id', 'item_id', 'wm_yr_wk']]
prices_df = pd.read_csv('../../../input/sell_prices.csv')
prices_df = encode_categorical(prices_df, ["item_id", "store_id"])


prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

print('calender')
calendar = pd.read_csv('../../../input/calendar.csv')
calendar = calendar[['wm_yr_wk', 'month', 'year']].drop_duplicates(subset=['wm_yr_wk'])
prices_df = prices_df.merge(calendar[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')

prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id', 'item_id', 'year'])['sell_price'].transform('mean')

prices_df.drop(['month', 'year', 'sell_price'], inplace=True, axis=1)
df_feat = df_all.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
df_feat.drop(['store_id', 'item_id', 'wm_yr_wk'], inplace=True, axis=1)


print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_price.pkl')
