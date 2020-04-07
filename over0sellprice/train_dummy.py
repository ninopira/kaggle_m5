# https://www.kaggle.com/kneroma/fnu050
from  datetime import datetime, timedelta
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

decide_x_feature = False

result_dir = './result/dummy/set_seed/baseline_shop_no_price_again_add_4weekdays_stat_std_shop/'
os.makedirs(result_dir, exist_ok=True)
print(result_dir)

wrmsse = True
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
    './feature/shop/f_diff_ave_sales_day_store_dept_std.pkl',
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
# sort
x_features = sorted(x_features)
print(x_features)
print('########################')
########################


########################
if wrmsse:
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
else:
    pass
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
# df_train = df_all.query('date <= "2016-03-27"')
# # val
# df_val = df_all.query('date > "2016-03-27" and date <= "2016-04-24"')
# # Todo: test1のみ
df_test = df_all.query('date > "2016-04-24" and date <= "2016-05-22"')
df_train = df_all.query('date <= "2016-04-24"')
print('train_shape{}'.format(df_train.shape))

t1 = time.time()
print('make_holdout:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('build_lgb_dataset')
t0 = time.time()
train_set = lgb.Dataset(df_train[x_features], df_train[target_col])
# val_set = lgb.Dataset(df_val[x_features], df_val[target_col])
fake_valid_inds = np.random.choice(len(df_train), 1000000)
fake_valid_data = lgb.Dataset(df_train[x_features].iloc[fake_valid_inds], label=df_train[target_col].iloc[fake_valid_inds])

t1 = time.time()
print('build_lgb_dataset:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

########################
print('########################')
print('learning..')
if wrmsse:
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
        valid_sets=[train_set, fake_valid_data],
        feval=wrmsse,
        verbose_eval=50)
else:
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
        num_boost_round=10,
        early_stopping_rounds=200,
        valid_sets=[train_set, fake_valid_data],
        verbose_eval=2)

del train_set, fake_valid_data

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
if wrmsse:
    val_WRMSSE = model.best_score['valid_1']['wrmsse']
else:
    val_WRMSSE = 999
print('WRMSSE:{}'.format(val_WRMSSE))
print('########################')
########################


def _create_fea(df_cnc):
    print('conc_test_:{}'.format(df_cnc.shape))
    # lagを埋める
    print('lag_7days_for_test_data...')
    t0 = time.time()
    df_cnc['demand_lag_7'] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7))
    print('check_lag7_mean:{}'.format(df_cnc['demand_lag_7'].mean()))
    for val in [7, 30, 60, 90, 180]:
        colname = f"demand_lag_7_roll_std_{val}"
        print(colname)
        df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(val).std())
        print(colname, 'mean', df_cnc[colname].mean())
    for val in [7, 30, 60, 90, 180]:
        colname = f"demand_lag_7_roll_mean_{val}"
        print(colname)
        df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(val).mean())
        print(colname, 'mean', df_cnc[colname].mean())
    colname = "demand_lag_7_roll_skew_30"
    print(colname)
    df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(30).skew())
    print(colname, 'mean', df_cnc[colname].mean())
    colname = "demand_lag_7_roll_kurt_30"
    print(colname)
    df_cnc[colname] = df_cnc.groupby(["id"])["demand"].transform(lambda x: x.shift(7).rolling(30).kurt())
    print(colname, 'mean', df_cnc[colname].mean())
    t1 = time.time()
    print('lag_7days_for_test_data:{0}'.format(t1-t0) + '[sec]')
    return df_cnc


alphas = [1.045, 1.035, 1.025, 1.015]
alphas = [1.000]
weights = [1/len(alphas)]*len(alphas)
sub = 0.

########################
print('########################')
print('pred...')
fday = datetime(2016, 4, 25)
max_lags = 220

for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
    cols = [f"F{i}" for i in range(1, 29)]
    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(icount, day)
        print(day - timedelta(days=max_lags))
        tst = df_all[(df_all.date >= day - timedelta(days=max_lags)) & (df_all.date <= day)].copy()
        print(tst.shape)
        _create_fea(tst)
        print(tst.shape)
        tst = tst.loc[tst.date == day, use_features]
        df_all.loc[df_all.date == day, "demand"] = alpha * model.predict(tst[x_features])

    te_sub = df_all.loc[df_all.date >= fday, ["id", "demand"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()["demand"][cols].reset_index()
    te_sub.fillna(0., inplace=True)
    te_sub.sort_values("id", inplace=True)
    te_sub.reset_index(drop=True, inplace=True)
    csv_path = os.path.join(result_dir, 'RMSE_{}_WRMSSE{}_count{}.csv'.format(val_RMSE, val_WRMSSE, icount))
    te_sub.to_csv(csv_path, index=False)
    if icount == 0:
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    print(icount, alpha, weight)

sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
csv_path = os.path.join(result_dir, 'RMSE_{}_WRMSSE{}.csv'.format(val_RMSE, val_WRMSSE))
print(sub.head())
sub.to_csv(csv_path, index=False)
