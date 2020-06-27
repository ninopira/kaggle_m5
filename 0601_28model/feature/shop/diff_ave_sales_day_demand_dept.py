import gc
import pandas as pd
import sys
sys.path.append('../../')


print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)

df_all = df_all[['id', 'date', 'store_id', 'dept_id', 'demand']]

for val in range(1, 29):
    print(val)
    df_all_tmp = df_all.copy()
    df_all_tmp[f'demand_lag_{val}'] = df_all_tmp.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
    print('groupby')
    # 日毎のdept_idのsell_priceの平均値
    df_day_store_dept_demand = df_all_tmp.groupby(['date', 'store_id', 'dept_id'])[f'demand_lag_{val}'].mean().reset_index()
    df_day_store_dept_demand.columns = ['date', 'store_id', 'dept_id', f'day_store_dept_lag{val}_demand_mean']

    df_new = pd.merge(df_all_tmp, df_day_store_dept_demand, on=['date', 'store_id', 'dept_id'], how='left')
    print(df_new.shape)

    # 同じdeptの商品の平均demandとの差 (高級な商品化を知りたい)
    df_new[f'diff_ave_demand_day_store_dept_lag{val}'] = (df_new[f'demand_lag_{val}'] - df_new[f'day_store_dept_lag{val}_demand_mean']) / df_new[f'day_store_dept_lag{val}_demand_mean']
    print(f'diff_ave_demand_day_store_dept_lag{val}', df_new[f'diff_ave_demand_day_store_dept_lag{val}'].mean())

    df_f = df_new.drop(['store_id', 'dept_id', 'demand', f'demand_lag_{val}'], axis=1)

    print(df_f.shape)
    print(df_f.head())
    print(df_f.tail())
    pkl_name = f'f_diff_ave_lag{val}demand_day_store_dept_no_rolling.pkl'
    print(pkl_name)
    df_f.to_pickle(pkl_name)

print('done')
