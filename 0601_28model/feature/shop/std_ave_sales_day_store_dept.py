import gc
import pandas as pd
import sys
import numpy as np
sys.path.append('../../')


print('read_transformed...')
df_all = pd.read_pickle('../../scaled_39804695_38950975_melt_over0sellprice.pkl')
print(df_all.shape)

print('groupby')
# 日毎のdept_idのsell_priceの平均値
df_day_store_dept_sales_price = df_all.groupby(['date', 'store_id', 'dept_id'])['sell_price'].agg(['mean', 'std']).reset_index()
df_day_store_dept_sales_price.columns = ['date', 'store_id', 'dept_id', 'day_store_dept_sales_mean_price', 'day_store_dept_sales_std_price']

df_new = pd.merge(df_all, df_day_store_dept_sales_price, on=['date', 'store_id', 'dept_id'], how='left')
print(df_new.shape)

# 同じdeptの商品の偏差値(高級な商品化を知りたい)
df_new['deviation_sales_day_store_dept'] = (df_new['sell_price'] - df_new['day_store_dept_sales_mean_price']) / df_new['day_store_dept_sales_std_price']
# 急激に安くなっていないか
df_new['lag_deviation_sales_day_store_dept'] = df_new.groupby(["id"])["deviation_sales_day_store_dept"].transform(lambda x: x.shift(1))


df_f = df_new[['id', 'date', 'day_store_dept_sales_std_price', 'deviation_sales_day_store_dept', 'lag_deviation_sales_day_store_dept']]

print(df_f.shape)
del df_all
gc.collect()

df_f.to_pickle('f_diff_ave_sales_day_store_dept_std.pkl')

print('done')
