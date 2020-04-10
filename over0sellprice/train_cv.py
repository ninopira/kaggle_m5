# https://www.kaggle.com/ragnar123/simple-lgbm-groupkfold-cv
import datetime
import gc
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import pandas as pd

# from wrmsse import bild_WRMSSEEvaluator, WRMSSEEvaluator_learge
from reduce_mem import reduce_mem_usage

use_top_importance = False
num_features = 50

result_dir = './result/set_seed/cv/baseline_shop_no_price_again_add_4weekdays_stat_std_shop_cumsum_zerodem/'
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
attrs = ["year", "month", "dayofweek", "is_year_end", "is_year_start", "week"]
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

if use_top_importance:
    csv_path = os.path.join(result_dir, 'importances.csv')
    df_importance = pd.read_csv(csv_path)
    df_importance.sort_values('gain', ascending=False, inplace=True)
    x_features = list(df_importance.head(num_features)['feature'])
    result_dir = os.path.join(result_dir, 'use_top_{}_importance_features'.format(num_features))
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)

use_features = x_features + [target_col] + ['id', 'date', 'week']
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
df_train = df_all.query('date <= "2016-04-24"')
# Todo: test1のみ
df_test = df_all.query('date > "2016-04-24" and date <= "2016-05-22"')

print('df_train:{}_df_test:{}'.format(df_train.shape, df_test.shape))
del df_all
gc.collect()
t1 = time.time()
print('make_holdout:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################


########################
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)
    hess = np.where(residual < 0, 2, 2 * 1.15)
    return grad, hess


# define custom evaluation metric
def custom_asymmetric_valid(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2), (residual ** 2) * 1.15)
    return "custom_asymmetric_eval", np.mean(loss), False
########################


########################
params = {
    'metric': ('custom', 'rmse'),
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'seed': 42,
    'learning_rate': 0.075,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'colsample_bytree': 0.85,
    'colsample_bynode': 0.85,
    'min_data_per_leaf': 25,
    'num_leaves': 200,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5}
########################

########################
print('########################')
print('learning..')
t0_all = time.time()
print('grouping..')
group = df_train['week'].astype(str) + '_' + df_train['year'].astype(str)
print('grouping_done.')
n_fold = 5
kf = GroupKFold(n_fold)
models = []
rmses = []
asymmetrics = []
importances_all = pd.DataFrame()
for fold, (trn_idx, val_idx) in enumerate(kf.split(df_train[x_features], df_train[target_col], group)):
    print(f'Training fold {fold + 1}')
    train_set = lgb.Dataset(df_train.iloc[trn_idx][x_features], df_train.iloc[trn_idx][target_col])
    val_set = lgb.Dataset(df_train.iloc[val_idx][x_features], df_train.iloc[val_idx][target_col])
    print('train:{}_val:{}'.format(df_train.iloc[trn_idx][x_features].shape, df_train.iloc[val_idx][x_features].shape))
    t0 = time.time()

    model = lgb.train(
        params,
        train_set,
        num_boost_round=5000,
        early_stopping_rounds=200,
        valid_sets=[train_set, val_set],
        feval=custom_asymmetric_valid,
        verbose_eval=50)
    # save_model
    models.append(model)
    model_path = os.path.join(result_dir, 'model_{}.lgb'.format(fold+1))
    model.save_model(model_path)
    t1 = time.time()
    print('Training fold {}:{}'.format(fold, t1-t0) + '[sec]')

    # metric
    val_RMSE = model.best_score['valid_1']['rmse']
    rmses.append(val_RMSE)
    val_asymmetric = model.best_score['valid_1']['custom_asymmetric_eval']
    asymmetrics.append(val_asymmetric)
    print('MSE:{}_ASYM{}'.format(val_RMSE, val_asymmetric))

    # pred
    y_pred = model.predict(df_test[x_features], num_iteration=model.best_iteration)
    df_test['demand'] += y_pred / n_fold

    # imprtance
    importances = pd.DataFrame()
    importances['feature'] = x_features
    importances['gain'] = model.feature_importance()
    importances['fold'] = fold + 1
    importances_all = pd.concat([importances_all, importances], axis=0, sort=False)

del train_set, val_set
gc.collect()

def save_importances(importances_: pd.DataFrame):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
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
print('all_learning:{}'.format(t1-t0_all) + '[sec]')
print('########################')
########################


########################
print('########################')
print('metric...')
val_RMSE = np.mean(rmses)
val_asymmetric = np.mean(asymmetrics)
print('RMSE:{}_ASYM{}'.format(val_RMSE, val_asymmetric))
print('########################')
########################


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
print('sub_shape:{}'.format(submission.shape))
csv_path = os.path.join(result_dir, 'RMSE:{}_ASYM{}.csv'.format(val_RMSE, val_asymmetric))
predict(df_test, submission, csv_path)
