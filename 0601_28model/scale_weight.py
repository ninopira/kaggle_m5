"""
id, ajust_weight, scaleのpandasの作成
データマートの目的変数をscale化して新しいpklとして書き出す
"""
import numpy as np
import pandas as pd

from metric import WRMSSEEvaluator


print('reading_dm.')
pkl_name = '35093990_33386550_melt_over0sellprice'
df_all = pd.read_pickle(f'./{pkl_name}.pkl')
print(df_all.shape)

print('reading_material')
calendar = pd.read_csv('../new_input/calendar.csv')
sales_train_validation = pd.read_csv('../new_input/sales_train_evaluation.csv')
sell_prices = pd.read_csv('../new_input/sell_prices.csv')

print('prepare_evaluator')
train_fold_df = sales_train_validation.copy()
valid_fold_df = sales_train_validation.iloc[:, -28:].copy()
# インスタンスの作成
evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)

# df_id = sales_train_validation.iloc[:,:6].copy()
print('calcurate_weight')
df_id = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
df_id['all_id'] = 0
for i in range(12):
    df_weight = getattr(evaluator, 'lv{}_weight'.format(i+1)).reset_index()  # evaluatorからweightを取得
    df_weight = df_weight.rename(columns={0: 'weight'})
    print(df_weight.head())

    # 各groupのsample数を計算
    group_keys = df_weight.columns[:-1].tolist()
    df_count = df_id.groupby(group_keys).count().id.reset_index()
    df_count = df_count.rename(columns={'id': 'n_sample'})

    df_weight = pd.merge(df_weight, df_count, how='left', on=(group_keys))
    assert df_weight.isnull().sum().sum() == 0, 'nullがあります'

    df_weight['weight_{}'.format(i+1)] = df_weight['weight'] / df_weight['n_sample']  # weightの計算
    df_id = pd.merge(df_id, df_weight.drop(['weight', 'n_sample'], axis=1), how='left', on=(group_keys))  # df_idにマージ

# scaleの計算
df_scale = getattr(evaluator, f'lv{12}_train_df').reset_index()[['item_id', 'store_id', 'scale']]
print(df_scale)
df_id = pd.merge(df_id, df_scale, how='left', on=(['item_id', 'store_id']))

# 最終weightの計算
df_id['ajust_weight'] = df_id.iloc[:, 7:19].mean(axis=1)*30490  # RMSSEはscaleにrootがあるので、rootをかけて補正

print(df_id.head())
print('df_id:', df_id.shape)

# 必要な列だけ抽出
df_scale_weeight = df_id[['id', 'scale', 'ajust_weight']]

print('merging_scale_and_scale_deemand')
print(df_all.shape)
df_all = pd.merge(df_all, df_scale_weeight[['id', 'scale']], how='left', on='id')
print(df_all.shape)
# scaleで割る 予測の時は掛けるのを忘れない
df_all['demand'] = df_all['demand'] / np.sqrt(df_all['scale'])
df_all = df_all.drop('scale', axis=1)

print('write...')
df_all.to_pickle(f'./scaled_{pkl_name}.pkl')
df_scale_weeight.to_pickle('scale_weight.pkl')

print('done')