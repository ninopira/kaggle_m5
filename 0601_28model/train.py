import datetime
import gc
import math
import os
import time

import click
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
import lightgbm as lgb
import pandas as pd

from reduce_mem import reduce_mem_usage
from metric import WRMSSEEvaluator

import warnings
warnings.filterwarnings('ignore')


@click.command()
@click.argument('days', type=int)
@click.option('--short_mode', is_flag=True)
def train(days, short_mode):
    print('*'*20, days, '*'*20)
    extract_test_day = datetime.datetime(2016, 5, 22) + datetime.timedelta(days=days)
    extract_test_day = extract_test_day.strftime('%Y-%m-%d')
    print('*'*20, extract_test_day, '*'*20)
    extract_test_old_day = datetime.datetime(2016, 4, 24) + datetime.timedelta(days=days)
    extract_test_old_day = extract_test_old_day.strftime('%Y-%m-%d')
    print('*'*20, extract_test_old_day, '*'*20)

    if short_mode:
        result_dir = f'./result/short/rm_ch_apply_cate_be_af/day{days}'
    else:
        result_dir = f'./result/rm_ch_apply_cate_be_af/day{days}'

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
    calendar = pd.read_csv('../new_input/calendar.csv')
    sales_train_validation = pd.read_csv('../new_input/sales_train_evaluation.csv')
    sell_prices = pd.read_csv('../new_input/sell_prices.csv')
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
        f'./feature/be_af_event/f_be_af.pkl'
        # cumsum'
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
        './feature/lag_demand/f_id_lag_demand_350_roll_28.pkl',
        # # lag sales
        './feature/lag_sales/f_id_lag_sales.pkl',
        # # shop
        f'./feature/shop/f_diff_ave_lag{days}demand_day_store_dept_no_rolling.pkl',
        f'./feature/shop/f_devine_ave_lag{days}demand_day_store_dept_no_roll.pkl',
        './feature/shop/f_diff_ave_sales_day_store_dept.pkl',
        './feature/shop/f_diff_ave_sales_day_store_dept_std.pkl',
        # trend_week
        f'./feature/trend_week/f_week_trend_{days}.pkl',
        # zero dem
        f'./feature/zero_demand/f_zero_demand_{days}.pkl',
        # 4weeks。未作成
        f'./feature/lag_4weeks/f_id_lag_demand_4weekdays_stat_{days}.pkl'
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

    print('rm_christomas...', df_all.shape)
    christomas_days = [datetime.datetime(2012, 12, 25), datetime.datetime(2013, 12, 25), datetime.datetime(2014, 12, 25), datetime.datetime(2015, 12, 25)]
    df_all = df_all[~df_all['date'].isin(christomas_days)]
    print(df_all.shape)


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
        },
        '3rd': {
            'train_end_date': '2016-04-24',
            'val_end_date': '2016-05-22',
            'train_end_date_num': 1913
        }
    }

    print('sep_test...')
    df_test = df_all.query('date > "2016-05-22"')
    df_test_old = df_all.query('date > "2016-04-24" and date <= "2016-05-22"')
    print('test_shape', df_test.shape, 'test_old_shape', df_test_old.shape)
    # https://www.kaggle.com/ejunichi/m5-three-shades-of-dark-darker-magic
    params = {
        'boosting_type': 'gbdt',
        # 'objective': 'poisson',
        'objective': 'rmse',
        # 'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.5,
        'subsample_freq': 1,
        'learning_rate': 0.07,
        'num_leaves': 2**11-1,
        'min_data_in_leaf': 2**12-1,
        'feature_fraction': 0.5,
        'max_bin': 100,
        'boost_from_average': False,
        'verbose': -1}

    print(params)

    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    # 元dfに対して予測して、wide_formatで返す関数
    def pred_and_convert_wide(df_features):
        pred_df = df_features[['id', 'date', 'demand']]
        # submission用に変換
        pred_df = pd.pivot(pred_df, index='id', columns='date', values='demand').reset_index()
        return pred_df

    wrmsse_score_list = []

    t0_all = time.time()
    print('train_main...')
    for num in ['1st', '2nd', '3rd']:
        print('########################')
        print('*'*20, num, '*'*20)
        t0 = time.time()
        df_test['demand'] = 0
        df_test_old['demand'] = 0
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
            valid_fold_df = sales_train_validation.iloc[:, -84:-56].copy()
        elif num == '2nd':
            train_fold_df = sales_train_validation.copy()  # weightの期間を変更
            valid_fold_df = sales_train_validation.iloc[:, -56:-28].copy()
        else:
            train_fold_df = sales_train_validation.copy()  # weightの期間を変更
            valid_fold_df = sales_train_validation.iloc[:, -28:].copy()

        train_set = lgb.Dataset(df_train[x_features], df_train[target_col], weight=ajust_weight_train, categorical_feature=cat_features)
        val_set = lgb.Dataset(df_val[x_features], df_val[target_col], weight=ajust_weight_val, categorical_feature=cat_features)
        print('start_learn')
        if short_mode:
            print('short_mode')
            model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=10,
                    early_stopping_rounds=200,
                    valid_sets=[train_set, val_set],
                    verbose_eval=2)
        else:
            model = lgb.train(
                    params,
                    train_set,
                    num_boost_round=5000,
                    early_stopping_rounds=200,
                    valid_sets=[train_set, val_set],
                    verbose_eval=50)
        # モデル書き出し
        model_path = os.path.join(result_dir, f'model_days{days}_val{num}.lgb')
        model.save_model(model_path)

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
        print('extract_val_day', df_val_extract['date'].unique(), df_val_extract.shape)
        val_pkl_path = os.path.join(result_dir, f'days{days}_val{num}.pkl')
        print(val_pkl_path)
        df_val_extract.to_pickle(val_pkl_path)

        # 予測。3つのcvに対して、それぞれtest, test_oldの予測を求める
        for df in [df_test, df_test_old]:
            y_pred = model.predict(df[x_features], num_iteration=model.best_iteration)
            df['demand'] = y_pred
            df = pd.merge(df, df_scale_weight[['id', 'scale']], how="left", on="id")
            # scaleを元に戻す
            df["demand"] = df["demand"] * np.sqrt(df["scale"])
            df = df[['id', 'date', 'demand']]

        # days後だけ取り出す
        print('extract_test')
        df_test_extract = df_test.query('date == @extract_test_day')
        print('extract_test_day', df_test_extract['date'].unique(), df_test_extract.shape)
        val_pkl_path = os.path.join(result_dir, f'days{days}_test_{num}.pkl')
        print(val_pkl_path)
        df_test_extract.to_pickle(val_pkl_path)
        print('extract_test_old')
        df_test_old_extract = df_test_old.query('date == @extract_test_old_day')
        print('extract_test_old_day', df_test_old_extract['date'].unique(), df_test_old_extract.shape)
        val_pkl_path = os.path.join(result_dir, f'days{days}_test_old_{num}.pkl')
        print(val_pkl_path)
        df_test_old_extract.to_pickle(val_pkl_path)

        # 重要度
        importances = pd.DataFrame()
        importances['feature'] = x_features
        importances['gain'] = model.feature_importance()

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

        # WRMSSEの算出
        if days == 28:
            # インスタンスの作成
            print('build_evaluater...')
            evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
            print('wrmsse...')
            id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            X_val_wide = pred_and_convert_wide(df_val)
            X_val_wide.columns = ['id'] + ['d_' + str(i + 1) for i in range(tr_val_date[num]['train_end_date_num'], tr_val_date[num]['train_end_date_num']+28)]
            valid_preds = pd.merge(train_fold_df[id_columns].copy(), X_val_wide, how="left", on="id")
            # スコアの算出
            wrmsse_score = evaluator.score(valid_preds.drop(id_columns, axis=1))
            wrmsse_score_list.append(wrmsse_score)
            print("WRMSSE：", round(wrmsse_score, 4))
            # all_valの書き出し
            val_pkl_path = os.path.join(result_dir, f'days{days}_val_all_{num}.pkl')
            print(val_pkl_path)
            df_val.to_pickle(val_pkl_path)

            # test書き出し
            print('write_test')
            def predict(test, submission, csv_path):
                predictions = test[['id', 'date', 'demand']]
                predictions = pd.pivot(predictions, index='id', columns='date', values='demand').reset_index()
                predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
                evaluation = submission[['id']].merge(predictions, on='id')

                validation_rows = [row for row in submission['id'] if 'validation' in row]
                validation = submission[submission['id'].isin(validation_rows)]
                final = pd.concat([validation, evaluation])
                print(final.head())
                print(final.tail())
                print(final.shape)
                final.to_csv(csv_path, index=False)

            submission = pd.read_csv('../new_input/sample_submission.csv')
            csv_path = os.path.join(result_dir, 'sub_28_{}_WRMSSE_{}.csv'.format(num, wrmsse_score))
            print(csv_path)
            predict(df_test, submission, csv_path)

        t1 = time.time()
        print('train_{}:{}'.format(num, t1-t0) + '[sec]')
        print('########################')
    t1 = time.time()
    print('train_all:{}'.format(t1-t0_all) + '[sec]')


if __name__ == '__main__':
    train()
