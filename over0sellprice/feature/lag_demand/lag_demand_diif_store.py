
import pandas as pd
import sys
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)

df_all = df_all[['id', 'date', 'store_id', 'dept_id', 'demand']]

# 日毎のdept_idのdemand平均値
df_day_store_dept_demand = df_all.groupby(['date', 'store_id', 'dept_id'])['demand'].agg(['mean', 'std']).reset_index()
df_day_store_dept_demand.columns = ['date', 'store_id', 'dept_id', 'day_store_dept_mean_demand', 'day_store_dept_std_demand']
df_new = pd.merge(df_all, df_day_store_dept_demand, on=['date', 'store_id', 'dept_id'], how='left')

# 同じdeptの商品の平均価格との差 (高級な商品化を知りたい)
df_new['diff_ave_demand_day_store_dept'] = (df_new['demand'] - df_new['day_store_dept_mean_demand']) / df_new['day_store_dept_mean_demand']
df_new['deviation_demand_day_store_dept'] = (df_new['demand'] - df_new['day_store_dept_mean_demand']) / df_new['day_store_dept_std_demand']


for val in [28, 29, 30]:
    colname = f"diff_ave_demand_day_store_dept_lag_{val}"
    df_new[colname] = df_new.groupby(["id"])["diff_ave_demand_day_store_dept"].transform(lambda x: x.shift(val))
    print(colname, df_new[colname] .mean())

    colname = f"deviation_demand_day_store_dept_lag_{val}"
    df_new[colname] = df_new.groupby(["id"])["deviation_demand_day_store_dept"].transform(lambda x: x.shift(val))
    print(colname, df_new[colname] .mean())

    colname = f"day_store_dept_mean_demand_lag_{val}"
    df_new[colname] = df_new.groupby(["id"])["day_store_dept_mean_demand"].transform(lambda x: x.shift(val))
    print(colname, df_new[colname] .mean())

    colname = f"day_store_dept_std_demand_lag_{val}"
    df_new[colname] = df_new.groupby(["id"])["day_store_dept_std_demand"].transform(lambda x: x.shift(val))
    print(colname, df_new[colname] .mean())

for val in [7, 30, 60, 90, 180]:
    colname = f"diff_ave_demand_day_store_dept_lag_28_roll_std_{val}"
    df_new[colname] = df_new.groupby(["id"])["diff_ave_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(val).std())
    print(colname, df_new[colname] .mean())

    colname = f"deviation_demand_day_store_dept_lag_28_roll_std_{val}"
    df_new[colname] = df_new.groupby(["id"])["deviation_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(val).std())
    print(colname, df_new[colname] .mean())

for val in [7, 30, 60, 90, 180]:
    colname = f"diff_ave_demand_day_store_dept_lag_28_roll_mean_{val}"
    df_new[colname] = df_new.groupby(["id"])["diff_ave_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(val).mean())
    print(colname, df_new[colname] .mean())

    colname = f"deviation_demand_day_store_dept_lag_28_roll_mean_{val}"
    df_new[colname] = df_new.groupby(["id"])["deviation_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(val).mean())
    print(colname, df_new[colname] .mean())

colname = "diff_ave_demand_day_store_dept_lag_28_roll_skew_30"
df_new[colname] = df_new.groupby(["id"])["diff_ave_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(30).skew())
print(colname, df_new[colname] .mean())

colname = "deviation_demand_day_store_dept_lag_28_roll_skew_30"
df_new[colname] = df_new.groupby(["id"])["deviation_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(30).skew())
print(colname, df_new[colname] .mean())


colname = "diff_ave_demand_day_store_dept_lag_28_roll_kurt_30"
df_new[colname] = df_new.groupby(["id"])["diff_ave_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(30).kurt())
print(colname, df_new[colname] .mean())

colname = "deviation_demand_day_store_dept_lag_28_roll_kurt_30"
df_new[colname] = df_new.groupby(["id"])["deviation_demand_day_store_dept"].transform(lambda x: x.shift(28).rolling(30).kurt())
print(colname, df_new[colname] .mean())


df_new.drop(['store_id', 'dept_id', 'demand',
             'day_store_dept_mean_demand', 'day_store_dept_std_demand',
             'diff_ave_demand_day_store_dept', 'deviation_demand_day_store_dept'],
            inplace=True, axis=1)

print(df_new.head())
print(df_new.shape)
df_new.to_pickle('f_id_lag_demand_diif_dev.pkl')
