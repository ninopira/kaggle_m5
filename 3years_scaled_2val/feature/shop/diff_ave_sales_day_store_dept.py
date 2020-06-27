import gc
import pandas as pd
import sys
sys.path.append('../../')


print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)

print('groupby')
# 日毎のdept_idのsell_priceの平均値
df_day_store_dept_sales_price = df_all.groupby(['date', 'store_id', 'dept_id'])['sell_price'].mean().reset_index()
df_day_store_dept_sales_price.columns = ['date', 'store_id', 'dept_id', 'day_store_dept_sales_mean_price']

df_new = pd.merge(df_all, df_day_store_dept_sales_price, on=['date', 'store_id', 'dept_id'], how='left')
print(df_new.shape)

# 同じdeptの商品の平均価格との差 (高級な商品化を知りたい)
df_new['diff_ave_sales_day_store_dept'] = (df_new['sell_price'] - df_new['day_store_dept_sales_mean_price']) / df_new['day_store_dept_sales_mean_price']
# 急激に安くなっていないか
df_new['lag_ave_sales_day_store_dept'] = df_new.groupby(["id"])["day_store_dept_sales_mean_price"].transform(lambda x: x.shift(1))
# 同じdeptの商品の平均価格が前日に比べどれくらい変化しているか(saleを見極めたい)
df_new['diff_lag_ave_sales_day_store_dept1'] = (df_new["day_store_dept_sales_mean_price"] - df_new["lag_ave_sales_day_store_dept"]) / (df_new["lag_ave_sales_day_store_dept"])

df_f = df_new[['id', 'date', 'day_store_dept_sales_mean_price', 'diff_ave_sales_day_store_dept', 'diff_lag_ave_sales_day_store_dept1']]

print(df_f.shape)
del df_all
gc.collect()

df_f.to_pickle('f_diff_ave_sales_day_store_dept.pkl')

print('done')
