"""
全量学習
fake validationか決めで止めるか
"""

import gc
import math
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
import lightgbm as lgb
import pandas as pd

from reduce_mem import reduce_mem_usage
from metric import WRMSSEEvaluator


fake_val = False
result_dir = './result/set_seed_all_train/fake_val_{}/baseline_shop_no_price_again_add_4weekdays_stat_std_shop_cumsum_zerodem_dem_shop_std_no_roll/'.format(fake_val)
os.makedirs(result_dir, exist_ok=True)
print(result_dir)


########################
print('########################')
# read_transfomed
print('read_transfomed_data')
t0 = time.time()
df_all = pd.read_pickle('./scaled_35093990_33386550_melt_over0sellprice.pkl')
df_all = reduce_mem_usage(df_all)
print(df_all.shape)
df_all = df_all.query('sell_price > 0')
print(df_all.shape)

print('read_scale_weight')
df_scale_weight = pd.read_pickle('./scale_weight.pkl')
print(df_scale_weight.shape)

t1 = time.time()
print('read_transfomed_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('read_other_data')
t0 = time.time()
calendar = pd.read_csv('../../input/calendar.csv')
sales_train_validation = pd.read_csv('../../input/sales_train_validation.csv')
sell_prices = pd.read_csv('../../input/sell_prices.csv')
t1 = time.time()
print('reead_other_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('merge_features...')
print('before_merged_shape:{}'.format(df_all.shape))
t0_all = time.time()

f_paths = [
    './feature/shop/f_devine_ave_lag28demand_day_store_dept_no_roll.pkl',
    './feature/shop/f_diff_ave_lag28demand_day_store_dept_no_rolling.pkl',
    './feature/zero_demand/f_id_zero_demand.pkl',
    './feature/cumsum/f_id_cumsum_demand.pkl',
    './feature/shop/f_diff_ave_sales_day_store_dept_std.pkl',
    './feature/lag_demand/f_id_lag_demand_4weekdays_stat.pkl',
    './feature/shop/f_diff_ave_sales_day_store_dept.pkl',
    './feature/lag_demand/f_id_lag_demand.pkl',
    './feature/lag_sales/f_id_lag_sales.pkl'
]

for f_path in f_paths:
    t0 = time.time()
    print(f_path)
    df_f = pd.read_pickle(f_path)
    reduce_mem_usage(df_f)
    print('feature_shape:{}'.format(df_f.shape))
    df_all = pd.merge(df_all, df_f, on=['id', 'date'], how='left')
    del df_f
    gc.collect()
    print('merged_shape:{}'.format(df_all.shape))
    t1 = time.time()
    print('merged_time:{0}'.format(t1-t0) + '[sec]')

print('all_merge_done')
t1 = time.time()
print('all_merged_time:{0}'.format(t1-t0_all) + '[sec]')
print('########################')
########################


########################
print('########################')
print('date_features...')
print('before_date_shape:{}'.format(df_all.shape))

df_all['date'] = pd.to_datetime(df_all['date'])
# 対象
attrs = ["year", "month", "dayofweek", "is_year_end", "is_year_start"]
# is_year_end, is_year_srart

for attr in attrs:
    dtype = np.int16 if attr == "year" else np.int8
    print(attr)
    df_all[attr] = getattr(df_all['date'].dt, attr).astype(dtype)
df_all["is_weekend"] = df_all["dayofweek"].isin([5, 6]).astype(np.int8)
print('add_date_shape:{}'.format(df_all.shape))
t1 = time.time()
print('date_feature:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
# setting_feature
print('df_all_col')
print(df_all.columns)
target_col = 'demand'
useless_cols = ['id', 'part',
                'date', 'wm_yr_wk', 'quarter', 'week', 'day',
                'is_quarter_end', 'is_quarter_start',
                'is_month_end', 'is_month_start',
                'release',
                # "is_year_end", "is_year_start"
                ]
# use: year, month, dayofweek, is_year_end, is_year_start, is_weekend
x_features = [col for col in df_all.columns if col not in list(useless_cols + [target_col])]

use_features = x_features + [target_col] + ['id', 'date']
x_features = list(set(x_features))

print('len_x_features:{}'.format(len(x_features)))
# sort
x_features = sorted(x_features)
print(x_features)

df_all = df_all[use_features]
print('rm_same_name_col')
print(df_all.shape)
df_all = df_all.loc[:, ~df_all.columns.duplicated()]
print(df_all.shape)
print('########################')


print('sep_test...')
# Note: 全量を学習に用いる

# Todo: test1のみ
df_train_tmp = df_all.query('date <= "2016-04-24"')
df_test = df_all.query('date > "2016-04-24" and date <= "2016-05-22"')

if fake_val:
    print('fake_val')
    fake_valid_inds = np.random.choice(df_train_tmp.index.values, 2_000_000, replace=False)
    train_inds = np.setdiff1d(df_train_tmp.index.values, fake_valid_inds)
    df_train = df_train_tmp.loc[train_inds]
    df_val = df_train_tmp.loc[fake_valid_inds]
    print('df_train:{}_df_val:{}'.format(df_train.shape, df_val.shape))
else:
    print('use_all')
    df_train = df_train_tmp.copy()
    print('df_train:{}'.format(df_train.shape))

del df_train_tmp
gc.collect()

params = {
    'metric': 'rmse',
    'objective': 'poisson',
    'n_jobs': -1,
    'seed': 20,
    'learning_rate': 0.05,
    'alpha': 0.1,
    'lambda': 0.1,
    'bagging_fraction': 0.66,
    'bagging_freq': 2,
    'colsample_bytree': 0.77}


# 元dfに対して予測して、wide_formatで返す関数
# def pred_and_convert_wide(df_features, x_features, model, df_scale=None):
#     pred_df = pd.DataFrame()
#     pred_df["demand"] = model.predict(df_features[x_features])
#     pred_df['id'] = list(df_features['id'])
#     pred_df['date'] = list(df_features['date'])

#     if df_scale is not None:
#         print('scaling_pred')
#         pred_df = pd.merge(pred_df, df_scale, how="left", on="id")
#         pred_df["demand"] = pred_df["demand"] * np.sqrt(pred_df["scale"])
#         pred_df = pred_df.drop("scale", axis=1)
#     print(pred_df.head())
#     # submission用に変換
#     pred_df = pd.pivot(pred_df, index='id', columns='date', values='demand').reset_index()
#     return pred_df


t0_all = time.time()
print('train_main...')

print('########################')
# weightを計算
print('caluculate_weight...')
train_fold_df = sales_train_validation.copy()  # weightの期間を変更
valid_fold_df = sales_train_validation.iloc[:, -28:].copy()


# インスタンスの作成
print('build_evaluater...')
evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)


print('start_learn')
evals_result = {}
ajust_weight_train = pd.merge(df_train['id'], df_scale_weight[['id', 'ajust_weight']], how="left", on="id")['ajust_weight']
train_set = lgb.Dataset(df_train[x_features], df_train[target_col], weight=ajust_weight_train)
if fake_val:
    ajust_weight_val = pd.merge(df_val['id'],  df_scale_weight[['id', 'ajust_weight']], how="left", on="id")['ajust_weight']
    val_set = lgb.Dataset(df_val[x_features], df_val[target_col], weight=ajust_weight_val)
    model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set, val_set],
            verbose_eval=50)
else:
    # 決めで止める
    model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set],
            verbose_eval=50)
# 書き出し
model_path = os.path.join(result_dir, f'model.lgb')
model.save_model(model_path)

# 予測
y_pred = model.predict(df_test[x_features], num_iteration=model.best_iteration)
df_test['demand'] += y_pred

# 重要度
importances = pd.DataFrame()
importances['feature'] = x_features
importances['gain'] = model.feature_importance()


def save_importances(importances_: pd.DataFrame):
    csv_path = os.path.join(result_dir, f'importances.csv')
    importances_.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 8))
    sns.barplot(
        x='gain',
        y='feature',
        data=importances_.sort_values('gain', ascending=False)[:50])
    png_path = os.path.join(result_dir, f'importances.png')
    plt.tight_layout()
    plt.savefig(png_path)


save_importances(importances)


# Warning: 全量学習なのでpass
# WRMSSEの算出
# print('wrmsse...')
# id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
# X_val_wide = pred_and_convert_wide(df_val, x_features, model, df_scale_weight[['id', 'scale']])
# X_val_wide.columns = ['id'] + ['d_' + str(i + 1) for i in range(tr_val_date[num]['train_end_date_num'], tr_val_date[num]['train_end_date_num']+28)]
# valid_preds = pd.merge(train_fold_df[id_columns].copy(), X_val_wide, how="left", on="id")
# valid_pred_df_list.append(valid_preds)
# # スコアの算出
# wrmsse_score = evaluator.score(valid_preds.drop(id_columns, axis=1))
# wrmsse_score_list.append(wrmsse_score)
# print("WRMSSE：", round(wrmsse_score, 4))
# t1 = time.time()
# print('train_{}:{}'.format(num, t1-t0) + '[sec]')
# print('########################')

t1 = time.time()
print('train_all:{}'.format(t1-t0_all) + '[sec]')

########################
print('########################')
print('post_process')

# scaleを元に戻す
pred_df = pd.merge(df_test, df_scale_weight[['id', 'scale']], how="left", on="id")
pred_df["demand"] = pred_df["demand"] * np.sqrt(pred_df["scale"])


# 書き出し
def predict(test, submission, csv_path):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index='id', columns='date', values='demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row]
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on='id')
    final = pd.concat([validation, evaluation])
    print(final.head())
    print(final.shape)
    final.to_csv(csv_path, index=False)


submission = pd.read_csv('../input/sample_submission.csv')
csv_path = os.path.join(result_dir, 'sub.csv')
predict(pred_df, submission, csv_path)
