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


# https://www.kaggle.com/mayer79/m5-forecast-poisson-loss
def prep_selling_prices(df):
    gr = df.groupby(["store_id", "item_id"])["sell_price"]
    df["sell_price_rel_diff"] = gr.pct_change()
    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())
    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())
    to_float32 = ["sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7"]
    df[to_float32] = df[to_float32].astype("float32")

    return df


prices_df = prep_selling_prices(prices_df)

df_feat = df_all.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
df_feat.drop(['store_id', 'item_id', 'wm_yr_wk', 'sell_price'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_price_2.pkl')
