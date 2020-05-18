"""
28dayの
- valを読み込んでcv値の算出
- テストデータを読み込んでsub.csvの作成
"""
import os
import time

import pandas as pd

from metric import WRMSSEEvaluator


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

# valの読み込み
print('val')
wrmsse_score_list = []
for num in ['1st', '2nd']:
    df_vals = pd.DataFrame()
    for days in range(1, 29):
        print('days', days)
        result_dir = f'./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/day{days}'
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
        valid_fold_df = sales_train_validation.iloc[:, -56:-28].copy()
    else:
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -28:].copy()
    print('build_evaluater...')
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
    df_vals = df_vals[['id', 'date', 'demand']]
    df_vals.to_pickle(f'./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/val_all_{num}.pkl')
    df_vals = pd.pivot(df_vals, index='id', columns='date', values='demand').reset_index()
    df_vals.columns = ['id'] + ['d_' + str(i + 1) for i in range(tr_val_date[num]['train_end_date_num'], tr_val_date[num]['train_end_date_num']+28)]
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    valid_preds = pd.merge(train_fold_df[id_columns].copy(), df_vals, how="left", on="id")
    wrmsse_score = evaluator.score(valid_preds.drop(id_columns, axis=1))
    wrmsse_score_list.append(wrmsse_score)
    print(num, "WRMSSE：", round(wrmsse_score, 4))

print(wrmsse_score_list)

print('test')
df_tests = pd.DataFrame()
for days in range(1, 29):
    print('days', days)
    result_dir = f'./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/day{days}'
    test_pkl_path = os.path.join(result_dir, f'days{days}_test.pkl')
    df_test_extract = pd.read_pickle(test_pkl_path)
    print(num, 'extract_day', df_test_extract['date'].unique(), df_test_extract.shape)

    if len(df_tests) == 0:
        df_tests = df_test_extract
    else:
        df_tests = pd.concat([df_tests, df_test_extract])

print('all_day_concat', df_tests['date'].unique(), len(df_tests['date'].unique()), df_vals.shape)


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


result_dir = f'./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/'

submission = pd.read_csv('../input/sample_submission.csv')
csv_path = os.path.join(result_dir, 'sub_inv_WRMSSE_{}_{}.csv'.format(wrmsse_score_list[0], wrmsse_score_list[1]))
predict(df_tests, submission, csv_path)






