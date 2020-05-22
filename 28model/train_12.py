import datetime
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


days = 12
extract_test_day = datetime.datetime(2016, 4, 24) + datetime.timedelta(days=days)
extract_test_day = extract_test_day.strftime('%Y-%m-%d')
print(extract_test_day)

result_dir = f'./result/28model/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag_more/day{days}'
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
    # cumsum
    f'./feature/cumsum/f_id_cumsum_demand_{days}.pkl',
    './feature/cumsum/f_id_cumsum_demand_90.pkl',
    './feature/cumsum/f_id_cumsum_demand_120.pkl',
    './feature/cumsum/f_id_cumsum_demand_180.pkl',
    './feature/cumsum/f_id_cumsum_demand_220.pkl',
    './feature/cumsum/f_id_cumsum_demand_364.pkl',
    # lag demnad
    f'./feature/lag_demand/f_id_demand_lag_{days}.pkl',
    f'./feature/lag_demand/f_id_demand_lag_{days+1}.pkl',
    f'./feature/lag_demand/f_id_demand_lag_{days+2}.pkl',
    f'./feature/lag_demand/f_id_lag_demand_{days}_roll.pkl',
    # lag sales
    './feature/lag_sales/f_id_lag_sales.pkl',
    # shop
    f'./feature/shop/f_diff_ave_lag{days}demand_day_store_dept_no_rolling.pkl',
    f'./feature/shop/f_devine_ave_lag{days}demand_day_store_dept_no_roll.pkl',
    './feature/shop/f_diff_ave_sales_day_store_dept.pkl',
    './feature/shop/f_diff_ave_sales_day_store_dept_std.pkl',
    # trend_week
    f'./feature/trend_week/f_week_trend_{days}.pkl',
    # zero dem
    f'./feature/zero_demand/f_zero_demand_{days}.pkl',
    # 4weeks
    f'./feature/lag4weeks/f_id_lag_demand_4weekdays_stat_{days}.pkl'
]
for day in [6, 7, 8, 13, 14, 15, 20, 21, 22]:
    if day != days:
        if days < day:
            f_paths.append(f'./feature/lag_demand/f_id_demand_lag_{day}.pkl')
            print('add', f'./feature/lag_demand/f_id_demand_lag_{day}.pkl')


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

########################
print('########################')
print('train')

tr_val_date = {
    '1st': {
        'train_end_date': '2016-02-28',
        'val_end_date': '2016-03-27',
        'train_end_date_num': 1857
    },
    '2nd': {
        'train_end_date': '2016-03-27',
        'val_end_date': '2016-04-24',
        'train_end_date_num': 1885
    }
}

print('sep_test...')
df_test = df_all.query('date > "2016-04-24" and date <= "2016-05-22"')

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
def pred_and_convert_wide(df_features):
    pred_df = df_features[['id', 'date', 'demand']]
    # submission用に変換
    pred_df = pd.pivot(pred_df, index='id', columns='date', values='demand').reset_index()
    return pred_df


model_list = []
imp_df_list = []
evaluator_list = []
wrmsse_score_list = []
eval_result_list = []

t0_all = time.time()
print('train_main...')
for num in ['1st', '2nd']:
    print('########################')
    t0 = time.time()
    train_end_date = tr_val_date[num]['train_end_date']
    val_end_date = tr_val_date[num]['val_end_date']
    print(f'val:{train_end_date}-{val_end_date}')
    df_train = df_all.query('date <= @train_end_date')
    df_val = df_all.query('date > @train_end_date and date <= @val_end_date')
    print('df_train:{}_df_val:{}'.format(df_train.shape, df_val.shape, ))
    # weightを計算
    print('caluculate_weight...')
    ajust_weight_train = pd.merge(df_train['id'], df_scale_weight[['id', 'ajust_weight']], how="left", on="id")['ajust_weight']
    ajust_weight_val = pd.merge(df_val['id'],  df_scale_weight[['id', 'ajust_weight']], how="left", on="id")['ajust_weight']

    if num == '1st':
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -56:-28].copy()
    else:
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -28:].copy()

    train_set = lgb.Dataset(df_train[x_features], df_train[target_col], weight=ajust_weight_train)
    val_set = lgb.Dataset(df_val[x_features], df_val[target_col], weight=ajust_weight_val)
    print('start_learn')
    evals_result = {}
    model = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            early_stopping_rounds=200,
            valid_sets=[train_set, val_set],
            verbose_eval=200)
    # モデル書き出し
    model_path = os.path.join(result_dir, f'model_days{days}_val{num}.lgb')
    model.save_model(model_path)

    # 予測
    y_pred = model.predict(df_test[x_features], num_iteration=model.best_iteration)
    df_test['demand'] += y_pred / 2.

    # 重要度
    importances = pd.DataFrame()
    importances['feature'] = x_features
    importances['gain'] = model.feature_importance()

    # validationへの予測
    df_val['demand'] = model.predict(df_val[x_features], num_iteration=model.best_iteration)
    df_val = pd.merge(df_val[['id', 'date', 'demand']], df_scale_weight[['id', 'scale']], how="left", on="id")
    df_val["demand"] = df_val["demand"] * np.sqrt(df_val["scale"])
    df_val = df_val.drop("scale", axis=1)

    # day日後だけ取り出す
    print('extract_val')
    extract_val_day = datetime.datetime.strptime(train_end_date, '%Y-%m-%d') + datetime.timedelta(days=days)
    extract_val_day = extract_val_day.strftime('%Y-%m-%d')
    print(extract_val_day)
    df_val_extract = df_val.query('date == @extract_val_day')
    print('extract_day', df_val_extract['date'].unique(), df_val_extract.shape)
    val_pkl_path = os.path.join(result_dir, f'days{days}_val{num}.pkl')
    print(val_pkl_path)
    df_val_extract.to_pickle(val_pkl_path)

    def save_importances(importances_: pd.DataFrame):
        csv_path = os.path.join(result_dir, f'{num}_importances.csv')
        importances_.to_csv(csv_path, index=False)
        plt.figure(figsize=(8, 8))
        sns.barplot(
            x='gain',
            y='feature',
            data=importances_.sort_values('gain', ascending=False)[:50])
        png_path = os.path.join(result_dir, f'days{days}_val{num}_importances.png')
        plt.tight_layout()
        plt.savefig(png_path)

    save_importances(importances)

    model_list.append(model)
    imp_df_list.append(importances)
    eval_result_list.append(evals_result)

    # WRMSSEの算出
    if days == 28:
        # インスタンスの作成
        print('build_evaluater...')
        evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
        evaluator_list.append(evaluator)
        print('wrmsse...')
        id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        X_val_wide = pred_and_convert_wide(df_val)
        X_val_wide.columns = ['id'] + ['d_' + str(i + 1) for i in range(tr_val_date[num]['train_end_date_num'], tr_val_date[num]['train_end_date_num']+28)]
        valid_preds = pd.merge(train_fold_df[id_columns].copy(), X_val_wide, how="left", on="id")
        # スコアの算出
        wrmsse_score = evaluator.score(valid_preds.drop(id_columns, axis=1))
        wrmsse_score_list.append(wrmsse_score)
        print("WRMSSE：", round(wrmsse_score, 4))
    t1 = time.time()
    print('train_{}:{}'.format(num, t1-t0) + '[sec]')
    print('########################')
t1 = time.time()
print('train_all:{}'.format(t1-t0_all) + '[sec]')

########################
print('########################')
print('post_process')

# scaleを元に戻す
df_test = pd.merge(df_test, df_scale_weight[['id', 'scale']], how="left", on="id")
df_test["demand"] = df_test["demand"] * np.sqrt(df_test["scale"])
df_test = df_test[['id', 'date', 'demand']]

# days後だけ取り出す
print('extract_test')
df_test_extract = df_test.query('date == @extract_test_day')
print('extract_day', df_test_extract['date'].unique(), df_test_extract.shape)
val_pkl_path = os.path.join(result_dir, f'days{days}_test.pkl')
print(val_pkl_path)
df_test_extract.to_pickle(val_pkl_path)

if days == 28:
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
    csv_path = os.path.join(result_dir, 'sub_28_WRMSSE_{}_{}.csv'.format(wrmsse_score_list[0], wrmsse_score_list[1]))
    predict(df_test, submission, csv_path)
