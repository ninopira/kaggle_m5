import datetime
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

decide_x_feature = False

result_dir = './result/7day/'
os.makedirs(result_dir, exist_ok=True)
print(result_dir)

########################
print('########################')
# read_transfomed
print('read_transfomed_data')
t0 = time.time()
df_all = pd.read_pickle('./23965140_22257700_melt.pkl')
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
    # './feature/price/f_price_no_rm_release.pkl',
    './feature/lag_demand/f_id_lag_demand_4weekdays_stat.pkl',
    './feature/shop/f_diff_ave_sales_day_store_dept.pkl',
    './feature/lag_demand/f_id_lag_demand_7day.pkl',
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

if decide_x_feature:
    x_features = [
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "event_name_1", "event_type_1",
        "snap_CA", "snap_TX", "snap_WI",
        "sell_price",
        # demand features.
        # "shift_t28", "rolling_std_t7", "rolling_std_t30", "rolling_std_t90", "rolling_std_t180",
        # "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60",
        'demand_lag_28', 'demand_lag_28_roll_std_7', 'demand_lag_28_roll_std_30', 'demand_lag_28_roll_std_90', 'demand_lag_28_roll_std_180',
        'demand_lag_28_roll_mean_7', 'demand_lag_28_roll_mean_30', 'demand_lag_28_roll_mean_60',

        # price features
        # "price_change_t1", "price_change_t365", "rolling_price_std_t7",
        'price_change_t1', 'price_change_t365', 'price_rolling_std_t7',
        # time features.
        "year", "month", "dayofweek"
        ]


use_features = x_features + [target_col] + ['id', 'date']
x_features = list(set(x_features))

print('len_x_features:{}'.format(len(x_features)))
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
    'metric': 'rmse',
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
    verbose_eval=50)

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
print('metric_MSE')
val_pred = model.predict(df_val[x_features], num_iteration=model.best_iteration)
val_MSE = np.sqrt(metrics.mean_squared_error(val_pred, df_val[target_col]))
print('MSE:{}'.format(val_MSE))
print('########################')
########################


########################
print('########################')
print('predict_test')
start_day = '2016-04-24'
df_test_preds = []
df_tr_val = pd.concat([df_train, df_val])
del df_train, df_val
gc.collect()
for i in range(4):
    end_day = datetime.datetime.strptime(start_day, '%Y-%m-%d') + datetime.timedelta(days=7)
    end_day = datetime.datetime.strftime(end_day, '%Y-%m-%d')
    print('start{}-end-{}'.format(start_day, end_day))

    # 7day分のみ取り出す
    df_test_small = df_test.query('date > @start_day & date <= @end_day')
    print('all_test_:{}'.format(df_test.shape))
    print('small_test_:{}'.format(df_test_small.shape))
    if i != 0:
        df_cnc = pd.concat([df_tr_val, df_test_used, df_test_small])
        print('conc_test_:{}'.format(df_cnc.shape))
        # lagを埋める
        print('lag_7days_for_test_data...')
        t0 = time.time()
        df_cnc['demand_lag_7'] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7))
        df_test_small = df_test_small.drop(['demand_lag_7'], axis=1).merge(df_cnc[['id', 'date', 'demand_lag_7']], on=['id', 'date'], how='left')
        print('check_lag7_mean:{}'.format(df_test_small['demand_lag_7'].mean()))
        for val in [7, 30, 60, 90, 180]:
            colname = f"demand_lag_7_roll_std_{val}"
            print(colname)
            df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(val).std())
            df_test_small = df_test_small.drop([colname], axis=1).merge(df_cnc[['id', 'date', colname]], on=['id', 'date'], how='left')
            print(colname, 'mean', df_test_small[colname].mean())
        for val in [7, 30, 60, 90, 180]:
            colname = f"demand_lag_7_roll_mean_{val}"
            print(colname)
            df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(val).mean())
            df_test_small = df_test_small.drop([colname], axis=1).merge(df_cnc[['id', 'date', colname]], on=['id', 'date'], how='left')
            print(colname, 'mean', df_test_small[colname].mean())
        colname = "demand_lag_7_roll_skew_30"
        print(colname)
        df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(30).skew())
        df_test_small = df_test_small.drop([colname], axis=1).merge(df_cnc[['id', 'date', colname]], on=['id', 'date'], how='left')
        print(colname, 'mean', df_test_small[colname].mean())
        colname = "demand_lag_7_roll_kurt_30"
        print(colname)
        df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(30).kurt())
        df_test_small = df_test_small.drop([colname], axis=1).merge(df_cnc[['id', 'date', colname]], on=['id', 'date'], how='left')
        print(colname, 'mean', df_test_small[colname].mean())

        t1 = time.time()
        print('lag_7days_for_test_data:{0}'.format(t1-t0) + '[sec]')
    # 予測
    print('pred...')
    df_test_small['demand'] = model.predict(df_test_small[x_features], num_iteration=model.best_iteration)
    df_test_preds.append(df_test_small)
    df_test_used = pd.concat(df_test_preds)
    # oldをnewに更新
    start_day = end_day
    df_test_small_old = df_test_small
df_test = pd.concat(df_test_preds)
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
csv_path = os.path.join(result_dir, 'RMSE_{}.csv'.format(val_MSE))
predict(df_test, submission, csv_path)