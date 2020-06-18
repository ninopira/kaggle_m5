import gc
import pandas as pd
import sys
sys.path.append('../../')


print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print(df_all.shape)

df_all = df_all[['id', 'date']]

df_all['pente'] = 0

pentecost_dates = [
    "2011-06-12", "2012-05-27", "2013-05-19", "2014-06-08",
    "2015-05-24", "2016-05-15",
]
orthodox_pentecost_dates = [
    "2011-06-12", "2012-06-03", "2013-06-23", "2014-06-08",
    "2015-05-31", "2016-06-19",
]

df_all.loc[df_all['date'].isin(pentecost_dates), ['pente']] = 1
df_all.loc[df_all['date'].isin(orthodox_pentecost_dates), ['pente']] = 2

print(df_all['pente'].value_counts())
print(df_all.shape)
print(df_all.head())
print(df_all.tail())

df_all.to_pickle('f_pente.pkl')

print('done')
