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

use_top_importance = False
num_features = 50

result_dir = './result/set_seed/baseline_shop_no_price_again_add_4weekdays_stat_std_shop/'
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

if use_top_importance:
    csv_path = os.path.join(result_dir, 'importances.csv')
    df_importance = pd.read_csv(csv_path)
    df_importance.sort_values('gain', ascending=False, inplace=True)
    x_features = list(df_importance.head(num_features)['feature'])
    result_dir = os.path.join(result_dir, 'use_top_{}_importance_features'.format(num_features))
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)


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
print('preparea_wrmse...')
t0 = time.time()
weight1, weight2, weight_mat_csr = weight_calc(df_all)


def wrmsse(preds, data):
    DAYS_PRED = 28
    NUM_ITEMS = 30490
    # this function is calculate for last 28 days to consider the non-zero demand period
    # actual obserbed values / 正解ラベル
    y_true = data.get_label()

    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED

    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(train[:, :num_col] - train[:, num_col:]), axis=1) / weight1) * weight2)
    return 'wrmsse', score, False


t1 = time.time()
print('preparea_wrmse:{0}'.format(t1-t0) + '[sec]')
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

print('df_train:{}_df_val:{}_df_test:{}'.format(df_train.shape, df_val.shape, df_test.shape))
del df_all
gc.collect()
t1 = time.time()
print('make_holdout:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('build_lgb_dataset')
t0 = time.time()
train_set = lgb.Dataset(df_train[x_features], df_train[target_col])
val_set = lgb.Dataset(df_val[x_features], df_val[target_col])

t1 = time.time()
print('build_lgb_dataset:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('learning..')
params = {
    'metric': ('custom', 'rmse'),
    'objective': 'poisson',
    'n_jobs': -1,
    'seed': 20,
    'learning_rate': 0.05,
    'alpha': 0.1,
    'lambda': 0.1,
    'bagging_fraction': 0.66,
    'bagging_freq': 2,
    'colsample_bytree': 0.77
    }

model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    early_stopping_rounds=200,
    valid_sets=[train_set, val_set],
    feval=wrmsse,
    verbose_eval=50)
model_path = os.path.join(result_dir, 'model.lgb')
model.save_model(model_path)
del train_set, val_set

importances = pd.DataFrame()
importances['feature'] = x_features
importances['gain'] = model.feature_importance()


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


########################
print('########################')
print('metric...')
val_RMSE = model.best_score['valid_1']['rmse']
print('MSE:{}'.format(val_RMSE))
val_WRMSSE = model.best_score['valid_1']['wrmsse']
print('WRMSSE:{}'.format(val_WRMSSE))
print('########################')
########################


########################
print('########################')
print('predict_test')

y_pred = model.predict(df_test[x_features], num_iteration=model.best_iteration)
df_test['demand'] = y_pred


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
csv_path = os.path.join(result_dir, 'RMSE_{}_WRMSSE{}.csv'.format(val_RMSE, val_WRMSSE))
predict(df_test, submission, csv_path)
