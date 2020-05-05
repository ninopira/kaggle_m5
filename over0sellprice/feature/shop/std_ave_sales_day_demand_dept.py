import gc
import pandas as pd
import sys
sys.path.append('../../')


print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)

df_all = df_all[['id', 'date', 'store_id', 'dept_id', 'demand']]

df_all['demand_lag_28'] = df_all.groupby(["id"])["demand"].transform(lambda x: x.shift(28))

print('groupby')
# 日毎のdept_idのsell_priceの平均値
df_day_store_dept_demand = df_all.groupby(['date', 'store_id', 'dept_id'])['demand_lag_28'].agg(['mean', 'std']).reset_index()
df_day_store_dept_demand.columns = ['date', 'store_id', 'dept_id', 'day_store_dept_sales_mean_lag28_demand', 'day_store_dept_sales_std_lag28_demand']

df_new = pd.merge(df_all, df_day_store_dept_demand, on=['date', 'store_id', 'dept_id'], how='left')
print(df_new.shape)

# 同じdeptの商品の平均demandとの差 (高級な商品化を知りたい)
df_new['devine_demand_day_store_dept_lag28'] = (df_new['demand_lag_28'] - df_new['day_store_dept_sales_mean_lag28_demand']) / df_new['day_store_dept_sales_std_lag28_demand']

for val in [7, 30, 60, 90, 180]:
    colname = f"devine_demand_day_store_dept_lag28_mean_{val}"
    df_new[colname] = df_new.groupby(['id'])["devine_demand_day_store_dept_lag28"].transform(lambda x: x.rolling(val).mean())
    print(colname, df_new[colname].mean())


df_f = df_new.drop(['store_id', 'dept_id', 'demand', 'demand_lag_28', 'day_store_dept_sales_mean_lag28_demand'], axis=1)

print(df_f.shape)
print(df_f.head())
print(df_f.tail())

df_f.to_pickle('f_devine_ave_lag28demand_day_store_dept.pkl')

print('done')
