import os
import time

import pandas as pd

from reduce_mem import reduce_mem_usage
from metric import WRMSSEEvaluator


# 元dfに対して予測して、wide_formatで返す関数
def pred_and_convert_wide(df_features):
    pred_df = df_features[['id', 'date', 'demand']]
    # submission用に変換
    pred_df = pd.pivot(pred_df, index='id', columns='date', values='demand').reset_index()
    return pred_df


t0 = time.time()
print('read_scale_weight')
df_scale_weight = pd.read_pickle('./scale_weight.pkl')
print(df_scale_weight.shape)
t1 = time.time()
print('read_scale_weight:{0}'.format(t1-t0) + '[sec]')

print('read_other_data')
t0 = time.time()
calendar = pd.read_csv('../../input/calendar.csv')
sales_train_validation = pd.read_csv('../../input/sales_train_validation.csv')
sell_prices = pd.read_csv('../../input/sell_prices.csv')
t1 = time.time()
print('reead_other_data:{0}'.format(t1-t0) + '[sec]')
print('########################')


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

wrmsse_score_list = []
for num in ['1st', '2nd']:
    # id, date, demand, pred
    df_val = pd.read_pickle(f'./result/28model/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/val_all_{num}.pkl')
    df_val_inv = pd.read_pickle(f'./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/val_all_{num}.pkl')
    df_val_final = df_val.copy()

    df_val = pd.merge(df_val,  df_scale_weight[['id', 'ajust_weight']], how="left", on="id")
    df_val_inv = pd.merge(df_val_inv,  df_scale_weight[['id', 'ajust_weight']], how="left", on="id")

    # weightを計算
    print('caluculate_weight...')
    ajust_weight_val = df_val['ajust_weight']
    # inv
    inv_ajust_weight_val = [1 / w for w in df_val_inv['ajust_weight']]

    # 重み付け
    df_val_final['demand'] = (df_val['demand'] * ajust_weight_val + df_val_inv['demand'] * inv_ajust_weight_val) / [we + inv for (we, inv) in zip(ajust_weight_val, inv_ajust_weight_val)]

    if num == '1st':
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -56:-28].copy()
    else:
        train_fold_df = sales_train_validation.copy()  # weightの期間を変更
        valid_fold_df = sales_train_validation.iloc[:, -28:].copy()
    print('build_evaluater...')
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
    print('wrmsse...')
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    X_val_wide = pred_and_convert_wide(df_val_final)
    X_val_wide.columns = ['id'] + ['d_' + str(i + 1) for i in range(tr_val_date[num]['train_end_date_num'], tr_val_date[num]['train_end_date_num']+28)]
    valid_preds = pd.merge(train_fold_df[id_columns].copy(), X_val_wide, how="left", on="id")
    # スコアの算出
    wrmsse_score = evaluator.score(valid_preds.drop(id_columns, axis=1))
    wrmsse_score_list.append(wrmsse_score)
    print("WRMSSE：", round(wrmsse_score, 4))

    df_val_final.to_pickle(f'./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/ems_val_{num}.pkl')

print(wrmsse_score_list)

df_test = pd.read_csv('./result/28model/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/sub_WRMSSE_0.6138112335680103_0.5409222306853285.csv')
df_test_inv = pd.read_csv('./result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/sub_inv_WRMSSE_0.6403738371412141_0.5505049640507121.csv')
df_test_final = df_test.copy()
print(df_test.shape, df_test_inv.shape)
print(len(df_test['id'].unique()), len(df_test_inv['id'].unique()))

# weight
df_test = pd.merge(df_test,  df_scale_weight[['id', 'ajust_weight']], how="left", on="id")
df_test_inv = pd.merge(df_test_inv,  df_scale_weight[['id', 'ajust_weight']], how="left", on="id")
print(df_test.shape, df_test_inv.shape, df_test_final.shape)

# weight
ajust_weight_test = df_test['ajust_weight']
inv_ajust_weight_test = [1 / w for w in df_test_inv['ajust_weight']]
# 足して1になるようにする
weight_test_sum = [we + inv for (we, inv) in zip(ajust_weight_test, inv_ajust_weight_test)]
ajust_weight_test = [we / we_sum for (we, we_sum) in zip(ajust_weight_test, weight_test_sum)]
inv_ajust_weight_test = [we / we_sum for (we, we_sum) in zip(inv_ajust_weight_test, weight_test_sum)]
# 重み付け
for col in ['F' + str(i + 1) for i in range(28)]:
    df_test_final[col] = df_test[col] * ajust_weight_test + df_test_inv[col] * inv_ajust_weight_test
    df_test_final[col].fillna(0, inplace=True)
    print(col)
    print('ori', df_test[col].mean())
    print('inv', df_test_inv[col].mean())
    print('final', df_test_final[col].mean())
print(df_test_final.head())
print(df_test_final.shape)

result_dir = './result/28model_inv/no_price_shop_cumsum_zerodem_dem_shop_std_week_trend_4weekstat_more_lag/'
submission = pd.read_csv('../input/sample_submission.csv')
print('ori_sub', submission.shape)
csv_path = os.path.join(result_dir, 'sub_ems_WRMSSE_{}_{}.csv'.format(wrmsse_score_list[0], wrmsse_score_list[1]))
df_test_final.to_csv(csv_path, index=False)