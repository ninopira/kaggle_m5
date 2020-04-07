import gc
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
import lightgbm as lgb
import pandas as pd

# from wrmsse import bild_WRMSSEEvaluator, WRMSSEEvaluator_learge
from reduce_mem import reduce_mem_usage
from wrmse import weight_calc

result_dir = './result/set_seed/baseline_shop_no_price_again_add_4weekdays_stat/av/'
os.makedirs(result_dir, exist_ok=True)
print(result_dir)

########################
print('########################')
# read_transfomed
print('read_transfomed_data')
t0 = time.time()
df_all = pd.read_pickle('./23965140_22257700_melt_over0sellprice.pkl')
df_all = reduce_mem_usage(df_all)
print(df_all.shape)
df_all = df_all.query('sell_price > 0')
print(df_all.shape)
t1 = time.time()
print('read_transfomed_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('merge_features...')
print('before_merged_shape:{}'.format(df_all.shape))
t0_all = time.time()

f_paths = [
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
print('########################')
########################


########################
print('########################')
print('make_holdout')
t0 = time.time()
df_all = df_all[use_features]
print('rm_same_name_col')
print(df_all.shape)
df_all = df_all.loc[:, ~df_all.columns.duplicated()]
print(df_all.shape)

print('sep...')
# train
# df_train = df_all[df_all['date'] <= '2016-03-27']
df_train = df_all.query('date <= "2016-03-27"')
# val
df_val = df_all.query('date > "2016-03-27" and date <= "2016-04-24"')
# Todo: test1のみ
df_test = df_all.query('date > "2016-04-24" and date <= "2016-05-22"')

# av
df_test[target_col] = 0
df_train[target_col] = 1
df_val[target_col] = 1
df_all = pd.concat([df_train, df_val, df_test], axis=0)

t1 = time.time()
print('make_holdout:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################


print('########################')
print('build_lgb_dataset')
t0 = time.time()
train_set = lgb.Dataset(df_all[x_features], df_all[target_col])
t1 = time.time()
print('build_lgb_dataset:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('learning..')
params = {
    'metric': 'auc',
    'objective': 'binary',
    'n_jobs': -1,
    'seed': 20,
    'learning_rate': 0.5,
    'alpha': 0.1,
    'lambda': 0.1,
    'bagging_fraction': 0.66,
    'bagging_freq': 2,
    'colsample_bytree': 0.77
    }

model = lgb.train(
    params,
    train_set,
    num_boost_round=100,
    early_stopping_rounds=5,
    valid_sets=[train_set],
    verbose_eval=10)

importances = pd.DataFrame()
importances['feature'] = x_features
importances['gain'] = model.feature_importance()
importances = importances.sort_values('gain', ascending=False)


def save_importances(importances_: pd.DataFrame):
    csv_path = os.path.join(result_dir, 'importances.csv')
    importances_.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 8))
    sns.barplot(
        x='gain',
        y='feature',
        data=importances_.sort_values('gain', ascending=False)[:50])
    png_path = os.path.join(result_dir, 'importances.png')
    plt.tight_layout()
    plt.savefig(png_path)


save_importances(importances)
t1 = time.time()
print('learning:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################
