"""
28dayの
- valを読み込んでcv値の算出
- テストデータを読み込んでsub.csvの作成
"""
import datetime
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from metric import WRMSSEEvaluator


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

# valの読み込み
print('val')
wrmsse_score_dict = {}
for num in ['1st', '2nd', '3rd']:
    print('########################')
    print('*'*20, num, '*'*20)
    df_vals = pd.DataFrame()
    for days in range(1, 29):
        print('days', days)
        result_dir = f'./result/rm_ch_apply_cate_be_af_rm_zero_poisson/day{days}'
        val_pkl_path = os.path.join(result_dir, f'days{days}_val{num}.pkl')
        df_val_extract = pd.read_pickle(val_pkl_path)
        print(num, 'extract_day', df_val_extract['date'].unique(), df_val_extract.shape)

        if len(df_vals) == 0:
            df_vals = df_val_extract
        else:
            df_vals = pd.concat([df_vals, df_val_extract])

    print('all_day_concat', df_vals['date'].unique(), len(df_vals['date'].unique()), df_vals.shape)

    if num == '1st':
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -84:-56].copy()
    elif num == '2nd':
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -56:-28].copy()
    else:
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -28:].copy()
    print('build_evaluater...')
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
    df_vals = df_vals[['id', 'date', 'demand']]
    df_vals = pd.pivot(df_vals, index='id', columns='date', values='demand').reset_index()
    df_vals.columns = ['id'] + ['d_' + str(i + 1) for i in range(tr_val_date[num]['train_end_date_num'], tr_val_date[num]['train_end_date_num']+28)]
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    valid_preds = pd.merge(train_fold_df[id_columns].copy(), df_vals, how="left", on="id")
    print(valid_preds.head())
    print(valid_preds.shape)
    wrmsse_score = evaluator.score(valid_preds.drop(id_columns, axis=1))
    wrmsse_score_dict[num] = wrmsse_score
    valid_preds.to_csv(f'./result/rm_ch_apply_cate_be_af_rm_zero_poisson/concat_val_{num}_{wrmsse_score}.csv', index=False)
    print(num, "WRMSSE：", round(wrmsse_score, 4))

print(wrmsse_score)

print('test')
for num in ['1st', '2nd', '3rd']:
    print('########################')
    print('*'*20, num, '*'*20)
    df_tests = pd.DataFrame()
    for days in range(1, 29):
        print('days', days)
        result_dir = f'./result/rm_ch_apply_cate_be_af_rm_zero_poisson/day{days}'
        test_pkl_path = os.path.join(result_dir, f'days{days}_test_{num}.pkl')
        df_test_extract = pd.read_pickle(test_pkl_path)[['id', 'date', 'demand']]
        print(num, 'extract_day', df_test_extract['date'].unique(), df_test_extract.shape)

        if len(df_tests) == 0:
            df_tests = df_test_extract
        else:
            df_tests = pd.concat([df_tests, df_test_extract])

    print('all_day_concat', df_tests['date'].unique(), len(df_tests['date'].unique()), df_vals.shape)

    # 何故かtrain.pyでscaleが元に戻っていないので対処療法
    df_scale_weight = pd.read_pickle('./scale_weight.pkl')
    df_tests = df_tests.merge(df_scale_weight, on=['id'], how='left')
    df_tests['demand'] = df_tests["demand"] * np.sqrt(df_tests["scale"])

    fig = plt.figure(figsize=(40, 16))
    ax = fig.add_subplot(1, 1, 1)
    df_tmp = df_tests.copy()
    df_tmp['date'] = pd.to_datetime(df_tmp['date'])
    df_tmp = df_tmp.groupby('date')[['demand']].sum().reset_index()
    for target in ['demand']:
        ax.plot(df_tmp['date'], df_tmp[target], label=target, marker="o")
        ax.legend(fontsize='20')
    ax.axvline(datetime.datetime(2016, 5, 30), color='blue', linestyle='--', alpha=0.4)  # Natioonal day
    ax.axvline(datetime.datetime(2016, 6, 2), color='green', linestyle='--', alpha=0.4)  # NBA
    ax.axvline(datetime.datetime(2016, 6, 7), color='orange', linestyle='--', alpha=0.4)  # ラマダン
    ax.axvline(datetime.datetime(2016, 6, 19), color='green', linestyle='--', alpha=0.4)  # NBA
    plt.savefig(f'./result/rm_ch_apply_cate_be_af_rm_zero_poisson/sub_{num}_WRMSSE_{wrmsse_score_dict[num]}.png')

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

    result_dir = f'./result/rm_ch_apply_cate_be_af_rm_zero_poisson/'
    submission = pd.read_csv('../input/sample_submission.csv')
    csv_path = os.path.join(result_dir, 'sub_{}_WRMSSE_{}.csv'.format(num, wrmsse_score_dict[num]))
    predict(df_tests, submission, csv_path)
